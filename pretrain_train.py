"""model.py 구조 + 사전학습 모델(KoCharELECTRA-small) 상위 층 학습 / 평가 / 추론.

augmented_train.py와 같은 흐름(분리 -> 학습 분할만 증강 -> 이전 오답 추가)에 모델만 pretrain_model.py로 바꾼 것.
입력과 배치 구성은 augmented_train.py와 완전히 같고, 처음부터 학습하는 층과 사전학습 층의 학습률만 따로 둔다.

학습:      python pretrain_train.py --aug_times 2
평가만:    python pretrain_train.py --eval_only   (저장된 체크포인트로 검증셋 평가 + test 추론)
오답 재학습: python pretrain_train.py --prev_error_path outputs_pretrain/prev_error.csv

결과물:
- checkpoints/best_pretrain.pt              검증 F1이 가장 높은 모델
- outputs_pretrain/val_predictions.csv      검증셋 ID, input, 정답, 예측, 문장별 F1 (F1 낮은 순)
- outputs_pretrain/prev_error.csv           학습/검증 분할에서 틀린 문장의 (예측, 정답) 쌍
- outputs_pretrain/submission.csv           test 예측 (sample_submission 형식)
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augmented_train import (
    ERROR_SUFFIX, augment_rows, evaluate, load_prev_errors, model_inputs, predict, set_seed, to_device,
)
from data_preprocessing import (
    IGNORE_INDEX, ObfuscationDataset, TokenBudgetSampler, Vocab, collate_fn, read_csv, split_rows, write_csv,
)
from pretrain_model import MODEL_NAME, PretrainDeobfuscationModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_path', default='data/train.csv')
    p.add_argument('--test_path', default='data/test.csv')
    p.add_argument('--ckpt_path', default='checkpoints/best_pretrain.pt')
    p.add_argument('--output_dir', default='outputs_pretrain')
    p.add_argument('--valid_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    # 증강 (augmented_train.py와 동일)
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
                   help='이전 학습에서 만든 prev_error.csv. 지정하면 (모델의 오답 문장 -> 정답) 쌍을 학습셋에 추가')
    # 모델
    p.add_argument('--model_name', default=MODEL_NAME)
    p.add_argument('--n_heads', type=int, default=8, help='하위(RoPE) 층의 head 수')
    p.add_argument('--n_layers', type=int, default=6, help='처음부터 학습하는 하위(RoPE) 층 수')
    p.add_argument('--n_pretrained_layers', type=int, default=4, help='위에 얹는 사전학습 모델의 상위 층 수 (최대 12)')
    p.add_argument('--d_ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1, help='하위 층과 분류 헤드 앞 dropout (사전학습 층 내부는 사전학습 설정 유지)')
    # 학습
    p.add_argument('--max_tokens', type=int, default=16384, help='배치당 (문장 수 x 최대 길이) 상한, GPU 메모리에 맞게 조절')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=5e-4, help='처음부터 학습하는 층(임베딩, 하위 층, 분류 헤드)의 학습률')
    p.add_argument('--pretrained_lr', type=float, default=1e-4, help='사전학습 층 중 최상위 층의 학습률')
    p.add_argument('--layer_decay', type=float, default=0.9,
                   help='사전학습 층에서 아래 층으로 갈수록 학습률에 곱하는 비율 (1이면 모든 층 동일)')
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


def param_groups(model, args):
    """처음부터 학습하는 층은 lr, 사전학습 층은 pretrained_lr에서 아래 층일수록 작은 학습률 (layer-wise lr decay).

    LayerNorm, bias, 임베딩은 weight decay 제외.
    """
    layers = model.pretrained.layer
    blocks = [(m, args.lr) for m in (model.embedding, model.layers, model.final_norm, model.head)]
    blocks += [(layer, args.pretrained_lr * args.layer_decay ** (len(layers) - 1 - i)) for i, layer in enumerate(layers)]
    groups = []
    for module, lr in blocks:
        params = [p for p in module.parameters() if p.requires_grad]
        matrix_decay = 0.0 if module is model.embedding else args.weight_decay
        for group, weight_decay in (([p for p in params if p.dim() >= 2], matrix_decay),
                                    ([p for p in params if p.dim() < 2], 0.0)):
            if group:
                groups.append({'params': group, 'lr': lr, 'weight_decay': weight_decay})
    assert sum(len(g['params']) for g in groups) == sum(p.requires_grad for p in model.parameters())
    return groups


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
        n_chars=len(vocab.chars), n_syllables=len(vocab.syllables),
        n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    )
    model = PretrainDeobfuscationModel(
        **model_config, n_pretrained_layers=args.n_pretrained_layers, model_name=args.model_name,
    ).to(device)
    # 체크포인트만으로 모델을 복원할 수 있도록 사전학습 층의 설정을 함께 저장
    model_config['electra_config'] = model.electra_config.to_dict()
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
    model = PretrainDeobfuscationModel(**ckpt['model_config']).to(device)
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
    train_set = ObfuscationDataset(train_rows, vocab)
    train_preds = predict(model, train_set, vocab, args, device)
    errors = [
        {'ID': f'{i}{ERROR_SUFFIX}', 'input': p, 'output': y}
        for dataset, dataset_preds in ((train_set, train_preds), (valid_set, preds))
        for i, y, p in zip(dataset.ids, dataset.outputs, dataset_preds) if p != y
    ]
    error_path = os.path.join(args.output_dir, 'prev_error.csv')
    write_csv(error_path, errors, ['ID', 'input', 'output'])
    print(f'오답 {len(errors)} / {len(train_set.inputs) + len(valid_set.inputs)}문장 (train + valid) -> {error_path}')

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
