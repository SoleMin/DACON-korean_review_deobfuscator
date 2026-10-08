"""2단계 교정 모델의 데이터셋. 1단계 예측문을 KcELECTRA 입력으로 바꾸고 구간 단위로 꺼낸다.

rows의 'input'은 난독화 문장, 'pred'는 1단계 예측문, 'output'은 정답이다.
KcELECTRA는 서브워드 모델이라, 1단계 예측문을 토크나이저로 나눈 뒤 글자마다 자기가 속한 토큰을 기록해 둔다.
토큰 위치가 512개까지라 510자를 넘는 문장은 겹치는 구간으로 나눠 넣고, 예측할 때 다시 이어 붙인다.
"""
import collections

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from data_preprocessing import IGNORE_INDEX, UNK, clean, is_hangul

AUG_SUFFIX = '_obf'  # double_augment.py가 추가 예측문의 ID에 붙이는 접미사 (뒤에 번호가 온다)

# 글자 단위 텐서의 패딩 값. 'obf'의 문자 id가 0(PAD)인 위치가 패딩이다
PADDING = {'obf': 0, 'pred': 0, 'char2tok': 0, 'copy_ids': -1, 'keep': False, 'labels': IGNORE_INDEX}


def split_chunks(n, window, overlap):
    """길이 n인 문장을 (start, end, keep_start, keep_end) 구간들로 나눈다.

    [start, end)는 모델에 넣는 범위, [keep_start, keep_end)는 그 구간의 예측을 쓰는 범위.
    keep 범위는 겹치는 부분의 가운데에서 갈라 서로 겹치지 않고 문장 전체를 덮는다.
    각 문자는 문맥이 더 넓게 보이는 구간의 예측을 쓰게 된다.
    """
    if n <= window:
        return [(0, n, 0, n)]
    assert 0 <= overlap < window, 'overlap은 max_chars보다 작아야 한다'
    # 마지막 구간은 문장 끝에 맞춰 모든 구간이 window 길이의 문맥을 갖게 한다
    starts = list(range(0, n - window, window - overlap)) + [n - window]
    cuts = [0] + [(a + window + b) // 2 for a, b in zip(starts, starts[1:])] + [n]
    return [(s, s + window, cuts[i], cuts[i + 1]) for i, s in enumerate(starts)]


class Stage2Dataset(Dataset):
    """길이/인덱스는 모두 구간 기준이다.

    train=True면 구간을 꺼낼 때마다 한글 위치의 일부를 무작위로 골라
    - mask_ratio만큼은 1단계 예측 글자를 가려, 난독화 자모와 문맥만으로 정답을 맞히게 하고
    - corrupt_ratio만큼은 1단계가 그 음절에서 실제로 냈던 오답으로 바꿔, 틀린 글자를 고치게 한다.
    매번 다른 위치가 바뀌므로 (실수 -> 정답) 예시가 epoch마다 새로 생긴다.
    """

    def __init__(self, rows, vocab, tokenizer, args, with_label=True, train=False):
        self.ids = [row['ID'] for row in rows]
        self.inputs = [row['input'] for row in rows]
        self.stage1 = [row['pred'] for row in rows]
        self.outputs = [clean(row['output']) for row in rows] if with_label else None
        self.vocab, self.tokenizer = vocab, tokenizer
        self.obf, self.copy_ids, self.labels, self.chunks = [], [], [], []  # chunks: (문장 idx, start, end, keep_start, keep_end)
        for i, (text, pred) in enumerate(zip(self.inputs, self.stage1)):
            assert len(text) == len(pred), f'{self.ids[i]}: 1단계 예측문의 길이가 input과 다르다'
            self.obf.append(torch.stack(vocab.encode_input(text), dim=1))  # (L, 4) 문자 id, 초성, 중성, 종성
            # 1단계가 예측한 음절의 출력 사전 id. 한글 위치가 아니거나 사전에 없는 음절이면 -1
            self.copy_ids.append(torch.tensor(
                [vocab.syl2id.get(p, -1) if is_hangul(x) else -1 for x, p in zip(text, pred)], dtype=torch.long,
            ))
            if with_label:
                self.labels.append(vocab.encode_label(text, self.outputs[i]))
            self.chunks += [(i, *chunk) for chunk in split_chunks(len(text), args.max_chars, args.overlap)]
        self.lengths = [end - start for _, start, end, _, _ in self.chunks]

        self.mask_ratio = args.mask_ratio if train else 0.0
        self.corrupt_ratio = args.corrupt_ratio if train else 0.0
        # 정답 음절 -> (1단계가 그 자리에 잘못 낸 음절들, 빈도). 이 데이터셋의 문장에서만 센다
        confusions = collections.defaultdict(collections.Counter)
        if self.corrupt_ratio > 0:
            for text, pred, out in zip(self.inputs, self.stage1, self.outputs):
                for x, p, y in zip(text, pred, out):
                    if is_hangul(x) and p != y:
                        confusions[y][p] += 1
        self.confusions = {
            y: (list(c.keys()), torch.tensor(list(c.values()), dtype=torch.float)) for y, c in confusions.items()
        }

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        sent, start, end, keep_start, keep_end = self.chunks[idx]
        n = end - start
        chars = list(self.stage1[sent][start:end])
        copy_ids = self.copy_ids[sent][start:end].clone()
        labels = self.labels[sent][start:end] if self.labels else torch.full((n,), IGNORE_INDEX, dtype=torch.long)
        masked = torch.zeros(n, dtype=torch.bool)
        if self.mask_ratio > 0 or self.corrupt_ratio > 0:
            target = labels != IGNORE_INDEX
            r = torch.rand(n)
            masked = target & (r < self.mask_ratio)
            corrupted = target & (r >= self.mask_ratio) & (r < self.mask_ratio + self.corrupt_ratio)
            answer = self.outputs[sent]
            for pos in corrupted.nonzero(as_tuple=True)[0].tolist():
                candidates = self.confusions.get(answer[start + pos])
                if candidates is None:  # 1단계가 한 번도 틀린 적 없는 음절은 그대로 둔다
                    continue
                wrong = candidates[0][torch.multinomial(candidates[1], 1).item()]
                chars[pos] = wrong
                copy_ids[pos] = self.vocab.syl2id.get(wrong, -1)

        text = ''.join(chars)
        pred = torch.stack(self.vocab.encode_input(text), dim=1)  # 1단계 예측 글자의 (문자 id, 초성, 중성, 종성)
        enc = self.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
        input_ids = torch.tensor(enc['input_ids'], dtype=torch.long)
        # 글자마다 자기가 속한 토큰의 위치. 공백처럼 어느 토큰에도 속하지 않는 글자는 0 ([CLS] 자리, 모델에서 0 벡터로 처리)
        char2tok = torch.zeros(n, dtype=torch.long)
        for tok, (s, e) in enumerate(enc['offset_mapping']):
            if e > s:  # 특수 토큰은 (0, 0)
                char2tok[s:e] = tok
        if masked.any():
            # 가린 글자: 그 글자가 속한 토큰을 [MASK]로 바꾸고, 글자 자체의 단서와 1단계 예측을 따라갈 단서도 지운다
            toks = char2tok[masked]
            input_ids[toks[toks > 0]] = self.tokenizer.mask_token_id
            pred[masked] = torch.tensor([UNK, 0, 0, 0])
            copy_ids[masked] = -1
        keep = torch.zeros(n, dtype=torch.bool)
        keep[keep_start - start:keep_end - start] = True
        return {'idx': idx, 'input_ids': input_ids, 'char2tok': char2tok, 'obf': self.obf[sent][start:end],
                'pred': pred, 'copy_ids': copy_ids, 'keep': keep, 'labels': labels,
                'pad': self.tokenizer.pad_token_id}


def collate_fn(batch):
    def pad(key, value):
        return pad_sequence([item[key] for item in batch], batch_first=True, padding_value=value)

    input_ids = pad('input_ids', batch[0]['pad'])
    n_tokens = torch.tensor([len(item['input_ids']) for item in batch])
    out = {
        'idx': [item['idx'] for item in batch],
        'input_ids': input_ids,
        'attention_mask': (torch.arange(input_ids.size(1))[None] < n_tokens[:, None]).long(),
    }
    out.update({key: pad(key, value) for key, value in PADDING.items()})
    return out


def model_inputs(batch, target_mask):
    return dict(
        input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], char2tok=batch['char2tok'],
        obf=batch['obf'], pred=batch['pred'], target_mask=target_mask, copy_ids=batch['copy_ids'][target_mask],
    )


def target_positions(batch):
    """예측할 위치: 이 구간이 맡은 범위(keep) 안의 한글 음절 (난독화 글자의 초성 > 0)."""
    return (batch['obf'][..., 1] > 0) & batch['keep']
