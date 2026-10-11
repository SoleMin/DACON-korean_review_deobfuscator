"""학습 분할에 적용하는 데이터 증강: 부분 복원, 단어 사전 치환, 재난독화, 이어 붙이기.
`augment_rows`가 네 가지를 모두 적용한다.

1) 부분 복원
모든 입력이 전부 난독화되어 있으면 자주 나오는 단어가 잘 학습되지 않는다. 문장의 절반을 무작위로 골라,
입력의 일부 단어(2개 ~ 문장 단어 수의 절반, 무작위 위치)를 원문으로 되돌린 복사본을 만든다.
정답은 그대로이므로, 모델은 이미 깨끗한 단어를 그대로 복사하는 것도 배운다.

2) 단어 사전 치환
단어 사전은 원문 단어마다 학습 분할에서 본 모든 난독화 형태를 모아 둔 것이다.
문장의 절반을 무작위로 골라, 입력의 각 단어를 `threshold`(기본 20%) 확률로
같은 원문 단어의 다른 난독화 형태로 바꾼 복사본을 만든다.
정답은 그대로이므로, 모델은 같은 단어의 새로운 난독화 형태를 보게 된다.

3) 재난독화
Obfuscator는 (입력, 정답) 쌍에서 자모 단위 난독화 규칙을 추정해 원문에 적용한다.
그래서 학습 데이터에 한 번도 나오지 않은 난독화 형태도 만들 수 있다.
문장의 절반을 무작위로 골라, 정답을 새로 난독화한 것을 입력으로 하는 복사본을 만든다.

4) 이어 붙이기
test 문장은 train 문장보다 훨씬 길다. 무작위로 고른 문장들을 공백으로 이어
긴 문장(기본 400~1600자)을 만들어 길이 분포를 맞춘다.

결과 = 원본 문장 전체 + 위 방법들로 만든 복사본.
"""
import collections
import random

from preprocessing import HANGUL_START, clean, decompose, is_hangul

# 종성 인덱스 -> 다음 음절로 넘어갈 때(연음) 되는 초성 인덱스 (겹받침은 뒤 자음)
JONG2CHO = {1: 0, 2: 1, 4: 2, 7: 3, 8: 5, 16: 6, 17: 7, 19: 9, 20: 10, 22: 12, 23: 14,
            24: 15, 25: 16, 26: 17, 3: 9, 5: 12, 9: 0, 10: 6, 11: 7, 12: 9, 13: 16, 14: 17, 18: 9}
NO_LIAISON = {0, 21, 27}  # 종성 없음, ㅇ, ㅎ
CHO_IEUNG = 11


def liaison_cho(prev_out):
    """앞 글자(원문)의 종성이 연음으로 넘어올 때의 초성. 넘어올 것이 없으면 None."""
    if not is_hangul(prev_out):
        return None
    jong = decompose(prev_out)[2]
    return None if jong in NO_LIAISON else JONG2CHO.get(jong)


class Obfuscator:
    """데이터에서 추정한 규칙으로 깨끗한 문장을 난독화한다. 출력 길이는 항상 입력 길이와 같다.

    - 음절마다 어느 자모를 바꿀지(초성/중성/종성 중 0~3개)를 관측된 분포에서 먼저 뽑는다.
      이때 그 음절에서 가능한 패턴만 후보로 둔다.
    - 바꾸기로 한 자모는 P(입력 자모 | 원문 자모, 바뀜)에서 뽑는다.
    - 연음: 초성을 바꾸는 경우, 원문 초성이 ㅇ이고 앞 음절에 종성이 있으면
      관측된 확률로 그 종성을 초성으로 끌어온다.
    - 한글이 아닌 문자는 그대로 복사한다.
    """

    def __init__(self, tables, patterns, p_liaison, seed=None):
        self.tables = tables      # [초성, 중성, 종성]: {원문 자모: (후보들, 가중치)}
        self.patterns = patterns  # {(초성, 중성, 종성 변경 여부): 빈도}
        self.p_liaison = p_liaison
        self.rng = random.Random(seed)

    @classmethod
    def from_rows(cls, rows, seed=None):
        """rows에서 규칙을 추정한다. 검증 문장이 섞이지 않도록 학습 분할만 넘겨야 한다."""
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
                    if p[k] != q[k] and not (k == 0 and used):  # 연음은 일반 치환으로 세지 않는다
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
            if not pats:  # 이 음절에 적용할 수 있는 규칙이 없음
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


def partially_deobfuscate(inp, out, rng):
    in_words, out_words = inp.split(' '), out.split(' ')
    if len(in_words) != len(out_words):  # 입력과 정답의 단어가 대응하지 않으면 건너뜀
        return None
    n = len(in_words)
    k = min(n, rng.randint(2, max(2, n // 2)))
    for i in rng.sample(range(n), k):
        in_words[i] = out_words[i]
    return ' '.join(in_words)


def substitute_from_dict(inp, out, variants, threshold, rng):
    in_words, out_words = inp.split(' '), out.split(' ')
    if len(in_words) != len(out_words):  # 입력과 정답의 단어가 대응하지 않으면 건너뜀
        return None
    changed = False
    for i, (in_word, out_word) in enumerate(zip(in_words, out_words)):
        if rng.random() >= threshold:
            continue
        # 길이가 같은 형태만: 입력과 정답은 글자 단위로 1:1 대응을 유지해야 한다
        candidates = [w for w in variants.get(out_word, ()) if w != in_word and len(w) == len(out_word)]
        if candidates:
            in_words[i] = rng.choice(candidates)
            changed = True
    return ' '.join(in_words) if changed else None  # 바뀐 단어가 없으면 원본과 같은 문장이 되므로 만들지 않음


def augment(rows, ratio, seed, word_dict=None, dict_ratio=0.5, threshold=0.2, obfuscator=None, obf_ratio=0.5):
    """문장마다 부분 복원(ratio), 단어 사전 치환(dict_ratio), 재난독화(obf_ratio) 복사본을 각 확률로 만든다."""
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
    """무작위 문장들을 공백으로 이어, 무작위 목표 길이에 닿을 때까지 붙인 긴 문장 n개를 만든다.

    test 문장이 train 문장보다 훨씬 길어서, 길이 분포를 맞추기 위한 것이다.
    입력과 정답의 길이가 다른 문장은 쓰지 않는다.
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


def build_word_dict(rows):
    """`augment`가 받는 형식의 단어 사전: [{'output': 원문 단어, 'input': [난독화 형태들]}, ...]"""
    variants = collections.defaultdict(set)
    for row in rows:
        in_words, out_words = row['input'].split(' '), clean(row['output']).split(' ')
        if len(in_words) != len(out_words):
            continue
        for in_word, out_word in zip(in_words, out_words):
            variants[out_word].add(in_word)
    return [{'output': out_word, 'input': sorted(in_words)} for out_word, in_words in variants.items()]


def augment_rows(rows, args):
    """rows에 증강 복사본을 더해 반환한다. 학습 분할만 넘겨야 한다.

    단어 사전과 난독화 규칙을 rows에서만 만들기 때문에 검증 문장의 정보가 섞이지 않는다.
    문장 단위 증강 세 가지는 seed를 바꿔 가며 aug_times번 적용하고, 이어 붙인 긴 문장은 마지막에 더한다.
    """
    word_dict = build_word_dict(rows)
    # Obfuscator는 호출할 때마다 다른 결과를 내므로 회차마다 새로 만들 필요가 없다
    obfuscator = Obfuscator.from_rows(rows, args.seed) if args.obf_ratio > 0 else None
    seen = {(row['input'], clean(row['output'])) for row in rows}
    aug_rows = []
    for i in range(args.aug_times):
        for row in augment(rows, args.aug_ratio, args.seed + i, word_dict, args.dict_ratio, args.threshold,
                           obfuscator, args.obf_ratio):
            key = (row['input'], row['output'])
            if key in seen:  # 원본이나 이전 회차와 완전히 같은 복사본은 제외
                continue
            seen.add(key)
            aug_rows.append({**row, 'ID': f"{row['ID']}_{i}"})
    rows = rows + aug_rows
    # test 문장이 훨씬 길기 때문에, 문장을 이어 붙인 긴 샘플을 더해 길이 분포를 맞춘다
    n_concat = int(len(rows) * args.concat_ratio)
    return rows + concat_rows(rows, n_concat, args.concat_min_len, args.concat_max_len, args.seed)
