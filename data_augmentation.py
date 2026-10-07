"""Data augmentation: partial de-obfuscation + word-dict substitution.

1) Partial de-obfuscation
Frequent words are poorly learned when every input is fully obfuscated. For a random
half of the rows, we create a copy whose input has some words restored to the original
text (2 words up to half of the words in the sentence, at random positions).
The output stays the same, so the model also learns to copy already-clean words.

2) Word-dict substitution
word_dict.jsonl maps each original word to every obfuscated form seen in train.
For a random half of the rows, we create a copy where each input word is swapped,
with probability `threshold` (default 20%), for another obfuscated form of the same
original word. The output stays the same, so the model sees new obfuscations of it.

3) Re-obfuscation
Obfuscator estimates jamo-level obfuscation rules from (input, output) pairs and applies
them to the original text, so it can produce obfuscated forms that never appear in train.
For a random half of the rows, we create a copy whose input is a fresh obfuscation of
the output.

Result = all original rows + the augmented copies of all methods.
"""
import argparse
import collections
import json
import random

from data_preprocessing import HANGUL_START, clean, decompose, is_hangul, read_csv, write_csv

# jong index -> cho index it becomes when carried over to the next syllable (last consonant of a cluster)
JONG2CHO = {1: 0, 2: 1, 4: 2, 7: 3, 8: 5, 16: 6, 17: 7, 19: 9, 20: 10, 22: 12, 23: 14,
            24: 15, 25: 16, 26: 17, 3: 9, 5: 12, 9: 0, 10: 6, 11: 7, 12: 9, 13: 16, 14: 17, 18: 9}
NO_LIAISON = {0, 21, 27}  # no jong, ㅇ, ㅎ
CHO_IEUNG = 11


def liaison_cho(prev_out):
    """cho that the jong of the previous original character would carry over, or None."""
    if not is_hangul(prev_out):
        return None
    jong = decompose(prev_out)[2]
    return None if jong in NO_LIAISON else JONG2CHO.get(jong)


class Obfuscator:
    """Obfuscates clean text with rules estimated from data. Output length always equals input length.

    - Per syllable, first sample which jamo to change (cho/jung/jong, 0~3 of them) from the observed
      distribution, restricted to the patterns that are possible for this syllable.
    - Each changed jamo is sampled from P(input jamo | original jamo, changed).
    - Liaison: when cho is changed, the original cho is ㅇ and the previous syllable has a jong,
      that jong is carried over as the new cho with the observed probability.
    - Non-Hangul characters are copied.
    """

    def __init__(self, tables, patterns, p_liaison, seed=None):
        self.tables = tables      # [cho, jung, jong]: {original jamo: (candidates, weights)}
        self.patterns = patterns  # {(cho, jung, jong changed or not): count}
        self.p_liaison = p_liaison
        self.rng = random.Random(seed)

    @classmethod
    def from_rows(cls, rows, seed=None):
        """Estimate the rules from rows. Pass the train split only, so validation rows do not leak in."""
        counts = [collections.defaultdict(collections.Counter) for _ in range(3)]
        patterns = collections.Counter()
        liaison_hit = liaison_total = 0
        for row in rows:
            inp, out = row['input'], clean(row['output'])
            if len(inp) != len(out):
                continue
            for i, (x, y) in enumerate(zip(inp, out)):
                if not is_hangul(x) or not is_hangul(y):
                    continue
                p, q = decompose(x), decompose(y)
                patterns[tuple(int(p[k] != q[k]) for k in range(3))] += 1
                carried = liaison_cho(out[i - 1]) if i > 0 and q[0] == CHO_IEUNG else None
                used = False
                if carried is not None and p[0] != q[0]:
                    liaison_total += 1
                    used = p[0] == carried
                    liaison_hit += used
                for k in range(3):
                    if p[k] != q[k] and not (k == 0 and used):  # liaison is not a plain substitution
                        counts[k][q[k]][p[k]] += 1
        tables = [{jamo: (list(c.keys()), list(c.values())) for jamo, c in table.items()} for table in counts]
        return cls(tables, dict(patterns), liaison_hit / max(liaison_total, 1), seed)

    def __call__(self, text):
        out = []
        for i, y in enumerate(text):
            if not is_hangul(y):
                out.append(y)
                continue
            q = list(decompose(y))
            carried = liaison_cho(text[i - 1]) if i > 0 and q[0] == CHO_IEUNG else None
            can = [q[k] in self.tables[k] or (k == 0 and carried is not None) for k in range(3)]
            pats = [(p, w) for p, w in self.patterns.items() if all(can[k] or not p[k] for k in range(3))]
            if not pats:  # no rule applies to this syllable
                out.append(y)
                continue
            pattern = self.rng.choices([p for p, _ in pats], [w for _, w in pats])[0]
            for k in range(3):
                if not pattern[k]:
                    continue
                if k == 0 and carried is not None and (
                    q[0] not in self.tables[0] or self.rng.random() < self.p_liaison
                ):
                    q[0] = carried
                else:
                    candidates, weights = self.tables[k][q[k]]
                    q[k] = self.rng.choices(candidates, weights)[0]
            out.append(chr(HANGUL_START + q[0] * 588 + q[1] * 28 + q[2]))
        return ''.join(out)


def read_word_dict(path):
    """jsonl -> list of dict: [{'output': original word, 'input': [obfuscated forms]}, ...]"""
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def partially_deobfuscate(inp, out, rng):
    in_words, out_words = inp.split(' '), out.split(' ')
    if len(in_words) != len(out_words):  # input/output should be aligned; skip if not
        return None
    n = len(in_words)
    k = min(n, rng.randint(2, max(2, n // 2)))
    for i in rng.sample(range(n), k):
        in_words[i] = out_words[i]
    return ' '.join(in_words)


def substitute_from_dict(inp, out, variants, threshold, rng):
    in_words, out_words = inp.split(' '), out.split(' ')
    if len(in_words) != len(out_words):  # input/output should be aligned; skip if not
        return None
    changed = False
    for i, (in_word, out_word) in enumerate(zip(in_words, out_words)):
        if rng.random() >= threshold:
            continue
        # same length only: input and output must stay aligned character by character
        candidates = [w for w in variants.get(out_word, ()) if w != in_word and len(w) == len(out_word)]
        if candidates:
            in_words[i] = rng.choice(candidates)
            changed = True
    return ' '.join(in_words) if changed else None  # no word swapped -> would duplicate the original row


def augment(rows, ratio, seed, word_dict=None, dict_ratio=0.5, threshold=0.2, obfuscator=None, obf_ratio=0.5):
    rng = random.Random(seed)
    variants = {d['output']: d['input'] for d in word_dict or []}
    aug = []
    for row in rows:
        out = clean(row['output'])
        if rng.random() < ratio:
            new_inp = partially_deobfuscate(row['input'], out, rng)
            if new_inp is not None:
                aug.append({'ID': f"{row['ID']}_aug", 'input': new_inp, 'output': out})
        if variants and rng.random() < dict_ratio:
            new_inp = substitute_from_dict(row['input'], out, variants, threshold, rng)
            if new_inp is not None:
                aug.append({'ID': f"{row['ID']}_dict", 'input': new_inp, 'output': out})
        if obfuscator is not None and rng.random() < obf_ratio:
            new_inp = obfuscator(out)
            if new_inp != row['input']:
                aug.append({'ID': f"{row['ID']}_obf", 'input': new_inp, 'output': out})
    return aug


def concat_rows(rows, n, min_len, max_len, seed):
    """Build n long rows by joining random rows with a space until a random target length is reached.

    Test sentences are much longer than train sentences, so this matches the length distribution.
    Rows whose input and output are not aligned are never used.
    """
    rng = random.Random(seed)
    pool = [(row['input'], clean(row['output'])) for row in rows]
    pool = [(inp, out) for inp, out in pool if len(inp) == len(out)]
    if not pool:
        return []
    result = []
    for i in range(n):
        target = rng.randint(min_len, max_len)
        inputs, outputs, length = [], [], 0
        while length < target:
            inp, out = rng.choice(pool)
            inputs.append(inp)
            outputs.append(out)
            length += len(inp) + 1
        result.append({'ID': f'concat_{i}', 'input': ' '.join(inputs), 'output': ' '.join(outputs)})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='data/train.csv')
    p.add_argument('--dst', default='data/train_aug.csv')  # not train.csv, so the original is never overwritten
    p.add_argument('--word_dict', default='data/word_dict.jsonl')
    p.add_argument('--ratio', type=float, default=0.5)  # share of rows that get a partially de-obfuscated copy
    p.add_argument('--dict_ratio', type=float, default=0.5)  # share of rows that get a dict-substituted copy
    p.add_argument('--threshold', type=float, default=0.2)  # share of words swapped in a dict-substituted copy
    p.add_argument('--obf_ratio', type=float, default=0.5)  # share of rows that get a re-obfuscated copy
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rows = read_csv(args.src)
    word_dict = read_word_dict(args.word_dict)
    obfuscator = Obfuscator.from_rows(rows, args.seed) if args.obf_ratio > 0 else None
    aug = augment(rows, args.ratio, args.seed, word_dict, args.dict_ratio, args.threshold, obfuscator, args.obf_ratio)
    write_csv(args.dst, [{'ID': r['ID'], 'input': r['input'], 'output': clean(r['output'])} for r in rows] + aug,
              ['ID', 'input', 'output'])
    n_dict = sum(r['ID'].endswith('_dict') for r in aug)
    n_obf = sum(r['ID'].endswith('_obf') for r in aug)
    print(
        f'{len(rows)} original + {len(aug) - n_dict - n_obf} de-obfuscated + {n_dict} dict-substituted'
        f' + {n_obf} re-obfuscated -> {args.dst}'
    )
