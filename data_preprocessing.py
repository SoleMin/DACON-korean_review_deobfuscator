"""데이터 로딩, 자모 분해, 인코딩, 토큰 예산 기반 배치 구성.

EDA 결과:
- input과 output은 (끝 공백 제외) 문자 단위로 1:1 정렬되어 있음
- 한글 음절이 아닌 문자(공백, 숫자, 문장부호 등)는 항상 그대로 복사됨
- 한글 음절은 항상 한글 음절로 바뀜
따라서 한글 음절 위치만 output 음절을 분류하고, 나머지는 input 문자를 그대로 복사한다.
"""
import csv
import random

import torch
from torch.utils.data import Dataset, Sampler

HANGUL_START, HANGUL_END = ord('가'), ord('힣')
N_CHO, N_JUNG, N_JONG = 19, 21, 28

# 비한글 문자 사전의 특수 토큰
PAD, UNK, HANGUL = 0, 1, 2
SPECIAL_CHARS = ['<pad>', '<unk>', '<hangul>']
IGNORE_INDEX = -100


def is_hangul(ch):
    return HANGUL_START <= ord(ch) <= HANGUL_END


def decompose(ch):
    """한글 음절을 (초성, 중성, 종성) 인덱스로 분해. 종성 0은 받침 없음."""
    code = ord(ch) - HANGUL_START
    return code // 588, code % 588 // 28, code % 28


def read_csv(path):
    # utf-8-sig: 파일 앞에 BOM이 있어도 첫 컬럼명이 'ID'로 읽히도록
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    # utf-8-sig: 엑셀에서 열어도 한글이 깨지지 않도록 BOM 추가
    with open(path, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clean(text):
    # output 끝에 공백이 붙은 행이 있어 끝 공백은 제거 (input과 길이를 맞추기 위함)
    return text.rstrip()


class Vocab:
    """input의 비한글 문자 사전과 output의 한글 음절 사전."""

    def __init__(self, chars, syllables):
        self.chars = SPECIAL_CHARS + sorted(chars)
        self.char2id = {c: i for i, c in enumerate(self.chars)}
        self.syllables = sorted(syllables)
        self.syl2id = {s: i for i, s in enumerate(self.syllables)}

    @classmethod
    def build(cls, rows):
        chars, syllables = set(), set()
        for row in rows:
            chars.update(c for c in row['input'] if not is_hangul(c))
            syllables.update(c for c in clean(row['output']) if is_hangul(c))
        return cls(chars, syllables)

    def state_dict(self):
        return {'chars': self.chars[len(SPECIAL_CHARS):], 'syllables': self.syllables}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state['chars'], state['syllables'])

    def encode_input(self, text):
        """문자열을 (문자 id, 초성, 중성, 종성) 텐서로 변환. 자모 인덱스 0은 비한글/패딩용."""
        n = len(text)
        char_ids = torch.empty(n, dtype=torch.long)
        cho = torch.zeros(n, dtype=torch.long)
        jung = torch.zeros(n, dtype=torch.long)
        jong = torch.zeros(n, dtype=torch.long)
        for i, ch in enumerate(text):
            if is_hangul(ch):
                c, v, t = decompose(ch)
                char_ids[i] = HANGUL
                cho[i], jung[i], jong[i] = c + 1, v + 1, t + 1
            else:
                char_ids[i] = self.char2id.get(ch, UNK)
        return char_ids, cho, jung, jong

    def encode_label(self, inp, out):
        """한글 위치만 output 음절 id, 나머지는 IGNORE_INDEX. 사전에 없는 음절도 IGNORE_INDEX."""
        labels = torch.full((len(inp),), IGNORE_INDEX, dtype=torch.long)
        for i, (x, y) in enumerate(zip(inp, out)):
            if is_hangul(x):
                labels[i] = self.syl2id.get(y, IGNORE_INDEX)
        return labels

    def decode(self, inp, hangul_pred_ids):
        """한글 위치는 순서대로 예측 음절로, 나머지는 input 문자 그대로 복사."""
        preds = iter(hangul_pred_ids)
        return ''.join(self.syllables[next(preds)] if is_hangul(x) else x for x in inp)


class ObfuscationDataset(Dataset):
    def __init__(self, rows, vocab, with_label=True):
        self.ids = [row['ID'] for row in rows]
        self.inputs = [row['input'] for row in rows]
        self.outputs = [clean(row['output']) for row in rows] if with_label else None
        # 매 epoch마다 인코딩하지 않도록 미리 텐서로 변환
        self.features = [vocab.encode_input(text) for text in self.inputs]
        self.labels = (
            [vocab.encode_label(i, o) for i, o in zip(self.inputs, self.outputs)] if with_label else None
        )
        self.lengths = [len(text) for text in self.inputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        item = {'idx': idx, 'features': self.features[idx]}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item


def collate_fn(batch):
    max_len = max(len(item['features'][0]) for item in batch)
    bsz = len(batch)
    char_ids = torch.full((bsz, max_len), PAD, dtype=torch.long)
    cho = torch.zeros(bsz, max_len, dtype=torch.long)
    jung = torch.zeros(bsz, max_len, dtype=torch.long)
    jong = torch.zeros(bsz, max_len, dtype=torch.long)
    labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=torch.long)
    for b, item in enumerate(batch):
        c, v, t, s = item['features']
        n = len(c)
        char_ids[b, :n], cho[b, :n], jung[b, :n], jong[b, :n] = c, v, t, s
        if 'labels' in item:
            labels[b, :n] = item['labels']
    return {
        'idx': [item['idx'] for item in batch],
        'char_ids': char_ids, 'cho': cho, 'jung': jung, 'jong': jong,
        'padding_mask': char_ids == PAD,
        'labels': labels,
    }


class TokenBudgetSampler(Sampler):
    """비슷한 길이끼리 묶고, (배치 크기 x 최대 길이)가 max_tokens를 넘지 않게 배치를 구성.

    길이 편차가 큰 데이터(최대 1381자, 평균 93자)에서 패딩 낭비를 줄이고,
    긴 문장이 들어와도 GPU 메모리가 일정하게 유지되도록 한다.
    """

    def __init__(self, lengths, max_tokens, shuffle=True, seed=42):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        self.batches = self._make_batches()

    def _make_batches(self):
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        if self.shuffle:
            # 길이 순서를 약간 섞어 매 epoch 배치 구성이 달라지게 함
            order = sorted(order, key=lambda i: self.lengths[i] + self.rng.uniform(-5, 5))
        batches, batch, max_len = [], [], 0
        for idx in order:
            new_max = max(max_len, self.lengths[idx])
            if batch and new_max * (len(batch) + 1) > self.max_tokens:
                batches.append(batch)
                batch, new_max = [], self.lengths[idx]
            batch.append(idx)
            max_len = new_max
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        if self.shuffle:
            self.batches = self._make_batches()
            self.rng.shuffle(self.batches)
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def split_rows(rows, valid_ratio, seed):
    rows = rows[:]
    random.Random(seed).shuffle(rows)
    n_valid = int(len(rows) * valid_ratio)
    return rows[n_valid:], rows[:n_valid]
