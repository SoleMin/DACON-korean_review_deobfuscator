"""학습 스크립트들이 함께 쓰는 도구: seed 고정, 문자 단위 F1, 학습 루프."""
import math
import random
import time

import torch
import torch.nn.functional as F

from data_preprocessing import IGNORE_INDEX


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


def mean_char_f1(preds, trues):
    return sum(char_f1(p, t) for p, t in zip(preds, trues)) / len(trues)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def fit(model, train_loader, optimizer, args, device, model_inputs, evaluate, save):
    """warmup + cosine 스케줄, fp16, EMA, 조기 종료를 적용해 학습하고 가장 높았던 검증 F1을 반환한다.

    검증 평가와 저장은 model이 아니라 가중치의 지수이동평균(EMA)으로 한다.
    - model_inputs(batch, target_mask): 배치에서 모델 입력(kwargs)을 만든다. 배치의 'labels'가 IGNORE_INDEX가 아닌 위치가 target이다
    - evaluate(ema_model): 검증 F1을 반환한다
    - save(ema_model, valid_f1, epoch): 검증 F1이 가장 높아질 때마다 불린다
    args에서 epochs, warmup_ratio, label_smoothing, ema_decay, patience, no_amp를 쓴다.
    """
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay),
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

        valid_f1 = evaluate(ema.module)
        improved = valid_f1 > best_f1
        print(
            f'epoch {epoch:3d} | loss {total_loss / max(total_tokens, 1):.4f} | valid char F1 {valid_f1:.4f}'
            f' | lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - start:.0f}s' + (' *' if improved else '')
        )
        if improved:
            best_f1, bad_epochs = valid_f1, 0
            save(ema.module, valid_f1, epoch)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f'조기 종료: {args.patience} epoch 동안 개선 없음')
                break
    return best_f1
