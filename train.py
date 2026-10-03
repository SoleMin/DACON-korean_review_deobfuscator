"""베이스라인 학습 / 평가 / 추론.

학습:      python train.py
평가만:    python train.py --eval_only   (저장된 체크포인트로 검증셋 평가 + test 추론)

결과물:
- checkpoints/best.pt               검증 F1이 가장 높은 모델
- outputs/val_predictions.csv       검증셋 ID, input, 정답, 예측, 문장별 F1 (F1 낮은 순)
- outputs/submission.csv            test 예측 (sample_submission 형식)
"""
import argparse
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_preprocessing import (
    IGNORE_INDEX, ObfuscationDataset, TokenBudgetSampler, Vocab, collate_fn, read_csv, split_rows, write_csv,
)
from model import DeobfuscationModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_path', default='data/train.csv')
    p.add_argument('--test_path', default='data/test.csv')
    p.add_argument('--ckpt_path', default='checkpoints/best.pt')
    p.add_argument('--output_dir', default='outputs')
    p.add_argument('--valid_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
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
    p.add_argument('--patience', type=int, default=7, help='검증 F1이 이 epoch 수 동안 오르지 않으면 조기 종료')
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


@torch.no_grad()
def predict(model, dataset, vocab, args, device):
    """데이터셋 전체를 예측해 문자열 리스트로 반환 (input 순서 유지)."""
    model.eval()
    loader = DataLoader(
        dataset, batch_sampler=TokenBudgetSampler(dataset.lengths, args.max_tokens * 2, shuffle=False),
        collate_fn=collate_fn, pin_memory=device.type == 'cuda',
    )
    preds = [None] * len(dataset)
    for batch in loader:
        batch = to_device(batch, device)
        target_mask = batch['cho'] > 0  # 한글 음절 위치
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda' and not args.no_amp):
            logits = model(**model_inputs(batch, target_mask))
        pred_ids = logits.argmax(-1).tolist()
        # 배치 전체에서 모은 한글 위치 예측을 문장별로 다시 나눔
        offset = 0
        for idx, n in zip(batch['idx'], target_mask.sum(1).tolist()):
            preds[idx] = vocab.decode(dataset.inputs[idx], pred_ids[offset:offset + n])
            offset += n
    return preds


def evaluate(model, dataset, vocab, args, device):
    preds = predict(model, dataset, vocab, args, device)
    f1s = [char_f1(p, t) for p, t in zip(preds, dataset.outputs)]
    return sum(f1s) / len(f1s), preds, f1s


def train(args, device):
    rows = read_csv(args.train_path)
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed)
    vocab = Vocab.build(train_rows)
    print(f'train {len(train_rows)} / valid {len(valid_rows)} | 비한글 문자 {len(vocab.chars)} | 출력 음절 {len(vocab.syllables)}')

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
            total_loss += loss.item() * labels.numel()
            total_tokens += labels.numel()

        valid_f1, _, _ = evaluate(model, valid_set, vocab, args, device)
        improved = valid_f1 > best_f1
        print(
            f'epoch {epoch:3d} | loss {total_loss / total_tokens:.4f} | valid char F1 {valid_f1:.4f}'
            f' | lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - start:.0f}s' + (' *' if improved else '')
        )
        if improved:
            best_f1, bad_epochs = valid_f1, 0
            torch.save(
                {'model': model.state_dict(), 'model_config': model_config, 'vocab': vocab.state_dict(),
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
    _, valid_rows = split_rows(read_csv(args.train_path), args.valid_ratio, args.seed)
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
