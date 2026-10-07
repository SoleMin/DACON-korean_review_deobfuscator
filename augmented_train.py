"""증강 데이터로 학습 / 평가 / 추론.

train.py와 동작은 같지만, train/valid를 먼저 나눈 뒤 train 쪽에만 증강을 적용한다.
(증강된 csv를 만들어 놓고 나누면 검증셋에 증강 결과물이 섞이고, 검증 문장의 증강본이 학습셋에 들어간다.)
단어 사전도 data/word_dict.jsonl(train 전체로 만든 것)을 쓰지 않고 학습 분할에서만 새로 만든다.

학습:      python augmented_train.py --aug_times 2
평가만:    python augmented_train.py --eval_only   (저장된 체크포인트로 검증셋 평가 + test 추론)
오답 재학습: python augmented_train.py --prev_error_path outputs_aug/prev_error.csv
           (이전 학습의 오답 문장을 input으로 한 데이터를 학습셋에 추가해 처음부터 다시 학습)

결과물:
- outputs_aug/prev_error.csv            학습/검증 분할에서 틀린 문장의 (예측, 정답) 쌍 (ID, input, output)
                                        검증 오답은 다음 학습에서 --seed를 바꿔 분할이 달라질 때만 학습에 쓰인다
- checkpoints/best_aug.pt               검증 F1이 가장 높은 모델
- outputs_aug/val_predictions.csv       검증셋 ID, input, 정답, 예측, 문장별 F1 (F1 낮은 순)
- outputs_aug/submission.csv            test 예측 (sample_submission 형식)
"""
import argparse
import math
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_augmentation import Obfuscator, augment, concat_rows
from data_preprocessing import (
    IGNORE_INDEX, ObfuscationDataset, TokenBudgetSampler, Vocab, clean, collate_fn, read_csv, split_rows, write_csv,
)
from model import DeobfuscationModel

ERROR_SUFFIX = '_err'  # prev_error.csv 행의 ID에 붙는 접미사

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_path', default='data/train.csv')
    p.add_argument('--test_path', default='data/test.csv')
    p.add_argument('--ckpt_path', default='checkpoints/best_aug.pt')
    p.add_argument('--output_dir', default='outputs_aug')
    p.add_argument('--valid_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    # 증강
    p.add_argument('--aug_times', type=int, default=1, help='학습셋에 증강을 적용해 추가하는 횟수 (0이면 증강 없음)')
    p.add_argument('--aug_ratio', type=float, default=0.5, help='1회당 부분 복원 증강본을 만드는 문장 비율')
    p.add_argument('--dict_ratio', type=float, default=0.5, help='1회당 단어 사전 치환 증강본을 만드는 문장 비율')
    p.add_argument('--threshold', type=float, default=0.2, help='사전 치환 증강본에서 단어가 치환될 확률')
    p.add_argument('--obf_ratio', type=float, default=0.5,
                   help='1회당 재난독화 증강본(정답을 추정한 규칙으로 새로 난독화)을 만드는 문장 비율 (0이면 끔)')
    p.add_argument('--concat_ratio', type=float, default=0.1,
                   help='문장을 이어 붙인 긴 샘플의 수 (증강까지 끝난 학습 문장 수 대비 비율, 0이면 끔)')
    p.add_argument('--concat_min_len', type=int, default=400, help='이어 붙인 샘플의 최소 목표 길이')
    p.add_argument('--concat_max_len', type=int, default=1600, help='이어 붙인 샘플의 최대 목표 길이')
    p.add_argument('--prev_error_path', default=None,
                   help='이전 학습에서 만든 prev_error.csv. 지정하면 (모델의 오답 문장 -> 정답) 쌍을 학습셋에 추가 (증강에는 쓰지 않음)')
    # 모델
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--d_ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    # 학습
    p.add_argument('--max_tokens', type=int, default=16384, help='배치당 (문장 수 x 최대 길이) 상한, GPU 메모리에 맞게 조절')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.05)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--ema_decay', type=float, default=0.999, help='가중치 EMA의 감쇠율 (0이면 EMA 없이 학습 중인 가중치 그대로)')
    p.add_argument('--patience', type=int, default=7, help='검증 F1이 이 epoch 수 동안 오르지 않으면 조기 종료')
    p.add_argument('--infer_window', type=int, default=0,
                   help='예측할 때 이보다 긴 문장은 이 길이의 겹치는 구간으로 나눠 넣는다 (0이면 문장 전체를 한 번에)')
    p.add_argument('--infer_overlap', type=int, default=64, help='구간으로 나눌 때 이웃 구간이 겹치는 문자 수')
    p.add_argument('--no_amp', action='store_true', help='fp16 혼합 정밀도 끄기')
    p.add_argument('--eval_only', action='store_true')
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def char_f1(pred, true):
    """문자 단위 F1: 같은 위치의 문자가 일치하는 개수로 precision/recall 계산."""
    if not pred and not true:
        return 1.0
    matches = sum(p == t for p, t in zip(pred, true))
    if matches == 0:
        return 0.0
    precision, recall = matches / len(pred), matches / len(true)
    return 2 * precision * recall / (precision + recall)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def model_inputs(batch, target_mask):
    return dict(
        char_ids=batch['char_ids'], cho=batch['cho'], jung=batch['jung'], jong=batch['jong'],
        padding_mask=batch['padding_mask'], target_mask=target_mask,
    )


def split_chunks(n, window, overlap):
    """길이 n인 문장을 (start, end, keep_start, keep_end) 구간들로 나눈다. window가 0이면 나누지 않는다.

    [start, end)는 모델에 넣는 범위, [keep_start, keep_end)는 그 구간의 예측을 쓰는 범위.
    keep 범위는 겹치는 부분의 가운데에서 갈라 서로 겹치지 않고 문장 전체를 덮는다.
    """
    if window <= 0 or n <= window:
        return [(0, n, 0, n)]
    assert 0 <= overlap < window, 'infer_overlap은 infer_window보다 작아야 한다'
    # 마지막 구간은 문장 끝에 맞춰 모든 구간이 window 길이의 문맥을 갖게 한다
    starts = list(range(0, n - window, window - overlap)) + [n - window]
    cuts = [0] + [(a + window + b) // 2 for a, b in zip(starts, starts[1:])] + [n]
    return [(s, s + window, cuts[i], cuts[i + 1]) for i, s in enumerate(starts)]


class ChunkDataset(Dataset):
    """ObfuscationDataset의 문장을 구간 단위로 꺼내는 예측용 데이터셋. 길이/인덱스는 모두 구간 기준."""

    def __init__(self, dataset, window, overlap):
        self.features = dataset.features
        # (문장 idx, start, end, keep_start, keep_end). 문장 순, 문장 안에서는 위치 순
        self.chunks = [(i, *chunk) for i, n in enumerate(dataset.lengths) for chunk in split_chunks(n, window, overlap)]
        self.lengths = [end - start for _, start, end, _, _ in self.chunks]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        sent, start, end, _, _ = self.chunks[idx]
        return {'idx': idx, 'features': tuple(f[start:end] for f in self.features[sent])}


@torch.no_grad()
def predict(model, dataset, vocab, args, device):
    """데이터셋 전체를 예측해 문자열 리스트로 반환 (input 순서 유지).

    infer_window가 0보다 크면 그보다 긴 문장은 겹치는 구간으로 나눠 예측하고,
    각 문자는 문맥이 더 넓게 보이는 구간의 예측을 가져와 다시 이어 붙인다.
    """
    model.eval()
    chunked = ChunkDataset(dataset, args.infer_window, args.infer_overlap)
    loader = DataLoader(
        chunked, batch_sampler=TokenBudgetSampler(chunked.lengths, args.max_tokens * 2, shuffle=False),
        collate_fn=collate_fn, pin_memory=device.type == 'cuda',
    )
    chunk_preds = [None] * len(chunked)
    for batch in loader:
        batch = to_device(batch, device)
        keep = torch.zeros_like(batch['padding_mask'])
        for b, idx in enumerate(batch['idx']):
            _, start, _, keep_start, keep_end = chunked.chunks[idx]
            keep[b, keep_start - start:keep_end - start] = True
        target_mask = (batch['cho'] > 0) & keep  # 이 구간이 맡은 한글 음절 위치
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda' and not args.no_amp):
            logits = model(**model_inputs(batch, target_mask))
        pred_ids = logits.argmax(-1).tolist()
        # 배치 전체에서 모은 한글 위치 예측을 구간별로 다시 나눔
        offset = 0
        for idx, n in zip(batch['idx'], target_mask.sum(1).tolist()):
            chunk_preds[idx] = pred_ids[offset:offset + n]
            offset += n
    # 구간이 문장 순, 위치 순이라 차례로 이어 붙이면 문장 전체 예측이 된다
    sent_preds = [[] for _ in dataset.inputs]
    for (sent, *_), ids in zip(chunked.chunks, chunk_preds):
        sent_preds[sent] += ids
    return [vocab.decode(text, ids) for text, ids in zip(dataset.inputs, sent_preds)]


def evaluate(model, dataset, vocab, args, device):
    preds = predict(model, dataset, vocab, args, device)
    f1s = [char_f1(p, t) for p, t in zip(preds, dataset.outputs)]
    return sum(f1s) / len(f1s), preds, f1s


def build_word_dict(rows):
    """data_augmentation.augment가 받는 형식의 단어 사전: [{'output': 원문 단어, 'input': [난독화 형태]}, ...]"""
    variants = defaultdict(set)
    for row in rows:
        in_words, out_words = row['input'].split(' '), clean(row['output']).split(' ')
        if len(in_words) != len(out_words):
            continue
        for in_word, out_word in zip(in_words, out_words):
            variants[out_word].add(in_word)
    return [{'output': out_word, 'input': sorted(in_words)} for out_word, in_words in variants.items()]


def augment_rows(rows, args):
    """증강을 aug_times회 적용해 원본 뒤에 붙인다. 회차마다 seed를 달리해 서로 다른 증강본이 나오게 한다."""
    word_dict = build_word_dict(rows)
    # 난독화 규칙도 학습 분할에서만 추정한다. 호출할 때마다 다른 결과가 나오므로 회차마다 새로 만들 필요는 없다
    obfuscator = Obfuscator.from_rows(rows, args.seed) if args.obf_ratio > 0 else None
    seen = {(row['input'], clean(row['output'])) for row in rows}
    aug_rows = []
    for i in range(args.aug_times):
        for row in augment(rows, args.aug_ratio, args.seed + i, word_dict, args.dict_ratio, args.threshold,
                           obfuscator, args.obf_ratio):
            key = (row['input'], row['output'])
            if key in seen:  # 원본이나 이전 회차와 완전히 같은 증강본은 제외
                continue
            seen.add(key)
            aug_rows.append({**row, 'ID': f"{row['ID']}_{i}"})
    rows = rows + aug_rows
    # test는 긴 문장의 비중이 훨씬 크므로, 문장을 이어 붙인 긴 샘플을 더해 길이 분포를 맞춘다
    n_concat = int(len(rows) * args.concat_ratio)
    return rows + concat_rows(rows, n_concat, args.concat_min_len, args.concat_max_len, args.seed)


def load_prev_errors(path, train_rows, valid_rows):
    """prev_error.csv에서 학습에 추가할 행만 고른다."""
    valid_ids = {row['ID'] for row in valid_rows}
    seen = {(row['input'], clean(row['output'])) for row in train_rows}
    error_rows, n_valid = [], 0
    for row in read_csv(path):
        # 현재 검증셋에 속한 문장의 오답은 제외 (넣으면 검증 문장의 정답을 학습하게 된다)
        if row['ID'].removesuffix(ERROR_SUFFIX) in valid_ids:
            n_valid += 1
            continue
        key = (row['input'], clean(row['output']))
        if key in seen:
            continue
        seen.add(key)
        error_rows.append(row)
    print(f'이전 오답 {len(error_rows)}개 추가 (현재 검증셋 문장이라 제외 {n_valid}개) <- {path}')
    return error_rows


def train(args, device):
    rows = read_csv(args.train_path)
    # 반드시 분리 먼저: 증강은 학습 분할에만 적용하고 검증셋은 원본 그대로 둔다
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed)
    n_original = len(train_rows)
    train_rows = augment_rows(train_rows, args)
    n_aug = len(train_rows) - n_original
    # 오답 데이터는 증강과 단어 사전이 끝난 뒤에 붙여서, 증강의 재료로 쓰이지 않게 한다
    error_rows = load_prev_errors(args.prev_error_path, train_rows, valid_rows) if args.prev_error_path else []
    train_rows = train_rows + error_rows
    vocab = Vocab.build(train_rows)
    print(
        f'train {len(train_rows)} (원본 {n_original} + 증강 {n_aug}, {args.aug_times}회 + 이전 오답 {len(error_rows)})'
        f' / valid {len(valid_rows)} | 비한글 문자 {len(vocab.chars)} | 출력 음절 {len(vocab.syllables)}'
    )

    train_set = ObfuscationDataset(train_rows, vocab)
    valid_set = ObfuscationDataset(valid_rows, vocab)
    train_loader = DataLoader(
        train_set, batch_sampler=TokenBudgetSampler(train_set.lengths, args.max_tokens, shuffle=True, seed=args.seed),
        collate_fn=collate_fn, pin_memory=device.type == 'cuda',
    )

    model_config = dict(
        n_chars=len(vocab.chars), n_syllables=len(vocab.syllables), d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    )
    model = DeobfuscationModel(**model_config).to(device)
    print(f'파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    # 가중치의 지수이동평균(EMA). 검증 평가와 체크포인트 저장은 model이 아니라 EMA 가중치로 한다
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay),
    )
    # LayerNorm, bias, 임베딩은 weight decay 제외
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and 'embedding' not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or 'embedding' in n]
    optimizer = torch.optim.AdamW(
        [{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}],
        lr=args.lr, betas=(0.9, 0.98),
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, 0.5 * (1 + math.cos(math.pi * min(step / total_steps, 1.0)))),
    )
    use_amp = device.type == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_f1, bad_epochs = -1.0, 0
    os.makedirs(os.path.dirname(args.ckpt_path) or '.', exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, total_loss, total_tokens = time.time(), 0.0, 0
        for batch in train_loader:
            batch = to_device(batch, device)
            target_mask = batch['labels'] != IGNORE_INDEX
            labels = batch['labels'][target_mask]
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(**model_inputs(batch, target_mask))
                loss = F.cross_entropy(logits.float(), labels, label_smoothing=args.label_smoothing)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update_parameters(model)
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

        valid_f1, _, _ = evaluate(ema.module, valid_set, vocab, args, device)
        improved = valid_f1 > best_f1
        print(
            f'epoch {epoch:3d} | loss {total_loss / total_tokens:.4f} | valid char F1 {valid_f1:.4f}'
            f' | lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - start:.0f}s' + (' *' if improved else '')
        )
        if improved:
            best_f1, bad_epochs = valid_f1, 0
            torch.save(
                {'model': ema.module.state_dict(), 'model_config': model_config, 'vocab': vocab.state_dict(),
                 'valid_f1': valid_f1, 'epoch': epoch},
                args.ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f'조기 종료: {args.patience} epoch 동안 개선 없음')
                break
    print(f'best valid char F1: {best_f1:.4f}')


def evaluate_and_export(args, device):
    ckpt = torch.load(args.ckpt_path, map_location=device)
    vocab = Vocab.from_state_dict(ckpt['vocab'])
    model = DeobfuscationModel(**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"체크포인트 로드: epoch {ckpt['epoch']}, valid F1 {ckpt['valid_f1']:.4f}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 검증셋 평가 결과 (학습 때와 같은 seed로 같은 검증셋을 재구성)
    train_rows, valid_rows = split_rows(read_csv(args.train_path), args.valid_ratio, args.seed)
    valid_set = ObfuscationDataset(valid_rows, vocab)
    valid_f1, preds, f1s = evaluate(model, valid_set, vocab, args, device)
    result = [
        {'ID': i, 'f1': round(f, 4), 'input': x, 'output': y, 'pred': p}
        for i, x, y, p, f in zip(valid_set.ids, valid_set.inputs, valid_set.outputs, preds, f1s)
    ]
    result.sort(key=lambda r: r['f1'])  # 틀린 문장부터 보이도록
    valid_path = os.path.join(args.output_dir, 'val_predictions.csv')
    write_csv(valid_path, result, ['ID', 'f1', 'input', 'output', 'pred'])
    print(f'valid char F1 {valid_f1:.4f} -> {valid_path}')

    # 한 글자라도 틀린 문장: 모델의 예측을 input으로, 정답을 output으로 저장 (학습 분할은 증강 전 원본만)
    # 검증셋 오답도 함께 저장한다. 다음 학습에서 검증셋에 속하는 문장은 load_prev_errors가 걸러내므로,
    # 다른 seed로 분할을 바꿔 돌릴 때만 학습에 쓰인다
    train_set = ObfuscationDataset(train_rows, vocab)
    train_preds = predict(model, train_set, vocab, args, device)
    train_errors = [
        {'ID': f'{i}{ERROR_SUFFIX}', 'input': p, 'output': y}
        for i, y, p in zip(train_set.ids, train_set.outputs, train_preds) if p != y
    ]
    valid_errors = [
        {'ID': f'{i}{ERROR_SUFFIX}', 'input': p, 'output': y}
        for i, y, p in zip(valid_set.ids, valid_set.outputs, preds) if p != y
    ]
    error_path = os.path.join(args.output_dir, 'prev_error.csv')
    write_csv(error_path, train_errors + valid_errors, ['ID', 'input', 'output'])
    print(
        f'오답 train {len(train_errors)} / {len(train_set)}문장 + valid {len(valid_errors)} / {len(valid_set)}문장'
        f' -> {error_path}'
    )

    # test 추론
    test_set = ObfuscationDataset(read_csv(args.test_path), vocab, with_label=False)
    test_preds = predict(model, test_set, vocab, args, device)
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    write_csv(submission_path, [{'ID': i, 'output': p} for i, p in zip(test_set.ids, test_preds)], ['ID', 'output'])
    print(f'test 예측 {len(test_preds)}개 -> {submission_path}')


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    if not args.eval_only:
        train(args, device)
    evaluate_and_export(args, device)


if __name__ == '__main__':
    main()
