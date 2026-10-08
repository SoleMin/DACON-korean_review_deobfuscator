"""2단계 교정 모델 학습 (1단계 예측 글자 가리기 / 일부러 틀리게 바꾸기 적용).

double_train.py와 같은 모델, 같은 평가/추론을 쓰고 학습용 데이터만 다르다.
학습 구간을 꺼낼 때마다 한글 위치의 일부를 무작위로 골라
- mask_ratio만큼은 1단계 예측 글자를 [MASK]로 가려, 난독화 자모와 문맥만으로 정답을 맞히게 하고
- corrupt_ratio만큼은 1단계가 그 음절에서 실제로 저질렀던 오답으로 바꿔, 틀린 글자를 고치게 한다.
매번 다른 위치가 바뀌므로 (실수 -> 정답) 예시가 epoch마다 새로 생긴다. 검증과 test 입력은 바꾸지 않는다.

먼저:      python double_stage1.py   (필요하면 python double_augment.py)
학습:      python double_mask_train.py --seed 44 --extra_oof_path outputs_double/stage1_oof_aug.csv
평가만:    python double_mask_train.py --eval_only

인자는 double_train.py의 것을 그대로 받고 --mask_ratio, --corrupt_ratio만 추가된다.
--ckpt_path, --output_dir을 주지 않으면 double_train.py의 결과를 덮어쓰지 않도록 아래 기본값을 쓴다.

결과물:
- checkpoints/double_stage2_mask.pt
- outputs_double_mask/val_predictions.csv, outputs_double_mask/submission.csv
"""
import argparse
import collections
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import double_train
from augmented_train import set_seed, to_device
from data_preprocessing import IGNORE_INDEX, TokenBudgetSampler, Vocab, is_hangul, read_csv, split_rows
from double_model import DoubleCorrectionModel, load_tokens
from double_train import (
    AUG_SUFFIX, DoubleDataset, collate_fn, evaluate, evaluate_and_export, model_inputs, param_groups, stage1_f1,
)

DEFAULT_CKPT_PATH = 'checkpoints/double_stage2_mask.pt'
DEFAULT_OUTPUT_DIR = 'outputs_double_mask'


def parse_args():
    """이 파일 전용 인자만 먼저 떼어 내고, 나머지는 double_train.parse_args에 그대로 넘긴다."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--mask_ratio', type=float, default=0.05, help='학습 때 1단계 예측 글자를 [MASK]로 가리는 한글 위치의 비율')
    p.add_argument('--corrupt_ratio', type=float, default=0.03,
                   help='학습 때 1단계 예측 글자를 1단계가 실제로 냈던 오답으로 바꾸는 한글 위치의 비율')
    extra, rest = p.parse_known_args()
    if not any(arg.startswith('--ckpt_path') for arg in rest):
        rest += ['--ckpt_path', DEFAULT_CKPT_PATH]
    if not any(arg.startswith('--output_dir') for arg in rest):
        rest += ['--output_dir', DEFAULT_OUTPUT_DIR]
    sys.argv = [sys.argv[0]] + rest
    args = double_train.parse_args()
    vars(args).update(vars(extra))
    return args


class CorruptedDoubleDataset(DoubleDataset):
    """꺼낼 때마다 1단계 예측 글자의 일부를 가리거나 오답으로 바꾸는 학습용 데이터셋."""

    def __init__(self, rows, vocab, token2id, args):
        super().__init__(rows, vocab, token2id, args)
        self.vocab, self.token2id = vocab, token2id
        self.unk, self.mask = token2id['[UNK]'], token2id['[MASK]']
        self.mask_ratio, self.corrupt_ratio = args.mask_ratio, args.corrupt_ratio
        # 정답 음절 -> (1단계가 그 자리에 잘못 낸 음절들, 빈도). 학습 데이터에서만 센다
        confusions = collections.defaultdict(collections.Counter)
        for text, pred, out in zip(self.inputs, self.stage1, self.outputs):
            for x, p, y in zip(text, pred, out):
                if is_hangul(x) and p != y:
                    confusions[y][p] += 1
        self.confusions = {y: (list(c.keys()), list(c.values())) for y, c in confusions.items()}

    def __getitem__(self, idx):
        item = super().__getitem__(idx)  # 매번 새 텐서라 여기서 고쳐도 원본 features는 그대로다
        sent, start, _, _, _ = self.chunks[idx]
        answer = self.outputs[sent]
        target = item['labels'] != IGNORE_INDEX
        r = torch.rand(len(target))
        masked = target & (r < self.mask_ratio)
        item['input_ids'][masked] = self.mask
        item['copy_ids'][masked] = -1  # 가린 자리에는 1단계 예측을 따라갈 단서를 주지 않는다
        corrupted = target & (r >= self.mask_ratio) & (r < self.mask_ratio + self.corrupt_ratio)
        for pos in corrupted.nonzero(as_tuple=True)[0].tolist():
            candidates = self.confusions.get(answer[start + pos - 1])  # [CLS] 때문에 1칸 밀려 있다
            if candidates is None:  # 1단계가 한 번도 틀린 적 없는 음절은 그대로 둔다
                continue
            wrong = candidates[0][torch.multinomial(torch.tensor(candidates[1], dtype=torch.float), 1).item()]
            item['input_ids'][pos] = self.token2id.get(wrong, self.unk)
            item['copy_ids'][pos] = self.vocab.syl2id.get(wrong, -1)
        return item


def train(args, device):
    """double_train.train과 같고, 학습용 데이터셋만 CorruptedDoubleDataset이다."""
    rows = read_csv(args.oof_path)
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed)
    n_original = len(train_rows)
    if args.extra_oof_path:
        # 검증 문장을 새로 난독화한 것은 뺀다 (넣으면 검증 문장의 정답을 학습하게 된다)
        valid_ids = {row['ID'] for row in valid_rows}
        train_rows = train_rows + [
            row for row in read_csv(args.extra_oof_path) if row['ID'].rsplit(AUG_SUFFIX, 1)[0] not in valid_ids
        ]
        print(f'추가 예측문 {len(train_rows) - n_original}개 <- {args.extra_oof_path}')
    vocab = Vocab.build(train_rows)
    tokens = load_tokens(args.model_name)
    token2id = {t: i for i, t in enumerate(tokens)}

    train_set = CorruptedDoubleDataset(train_rows, vocab, token2id, args)
    valid_set = DoubleDataset(valid_rows, vocab, token2id, args)
    base_f1s = stage1_f1(valid_set)
    print(
        f'train {len(train_rows)} / valid {len(valid_rows)} | 구간 수 train {len(train_set)} / valid {len(valid_set)}'
        f' | 출력 음절 {len(vocab.syllables)} | 교정 전 valid char F1 {sum(base_f1s) / len(base_f1s):.4f}'
    )
    print(
        f'가리기 {args.mask_ratio:g} / 오답으로 바꾸기 {args.corrupt_ratio:g}'
        f' (1단계가 틀린 적 있는 음절 {len(train_set.confusions)}종)'
    )
    train_loader = DataLoader(
        train_set, batch_sampler=TokenBudgetSampler(train_set.lengths, args.max_tokens, shuffle=True, seed=args.seed),
        collate_fn=collate_fn, pin_memory=device.type == 'cuda',
    )

    model = DoubleCorrectionModel(
        len(vocab.syllables), model_name=args.model_name, dropout=args.dropout, copy_bias=args.copy_bias,
    ).to(device)
    # 체크포인트만으로 모델을 복원할 수 있도록 인코더 설정을 함께 저장
    model_config = dict(
        n_syllables=len(vocab.syllables), electra_config=model.electra.config.to_dict(), dropout=args.dropout,
    )
    print(f'파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    # 가중치의 지수이동평균(EMA). 검증 평가와 체크포인트 저장은 model이 아니라 EMA 가중치로 한다
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay),
    )

    optimizer = torch.optim.AdamW(param_groups(model, args), lr=args.lr, betas=(0.9, 0.999), eps=1e-6)
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
            if labels.numel() == 0:  # 한글이 하나도 없는 배치: loss가 nan이 되므로 건너뜀
                scheduler.step()
                continue
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

        # 검증은 가리지 않은 입력으로 한다 (loss는 가린 입력 기준이라 double_train.py보다 높게 나온다)
        valid_f1, _, _ = evaluate(ema.module, valid_set, vocab, args, device)
        improved = valid_f1 > best_f1
        print(
            f'epoch {epoch:3d} | loss {total_loss / max(total_tokens, 1):.4f} | valid char F1 {valid_f1:.4f}'
            f' | lr {scheduler.get_last_lr()[-1]:.2e} | {time.time() - start:.0f}s' + (' *' if improved else '')
        )
        if improved:
            best_f1, bad_epochs = valid_f1, 0
            torch.save(
                {'model': ema.module.state_dict(), 'model_config': model_config, 'vocab': vocab.state_dict(),
                 'tokens': tokens, 'valid_ids': valid_set.ids, 'valid_f1': valid_f1, 'epoch': epoch},
                args.ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f'조기 종료: {args.patience} epoch 동안 개선 없음')
                break
    print(f'best valid char F1: {best_f1:.4f} (교정 전 {sum(base_f1s) / len(base_f1s):.4f})')


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
