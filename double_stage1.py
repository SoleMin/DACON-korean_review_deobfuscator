"""2단계 교정 파이프라인의 1단계: K-fold 학습으로 '처음 보는 문장에 대한 예측문'을 만든다.

2단계 교정 모델(double_train.py)은 1단계 모델이 실제로 저지르는 실수를 보고 배워야 한다.
그런데 1단계 모델은 자기가 학습한 문장은 거의 다 맞히므로, train을 n_folds 조각으로 나눠
(나머지 조각으로 학습 -> 안 본 조각을 예측)을 조각마다 반복해 train 전체의 예측문을 모은다.
test는 조각별 모델들의 확률을 평균해(앙상블) 예측한다.

학습 흐름(분리 -> 학습 분할만 증강)과 모델은 augmented_train.py / pretrain_train.py와 같다.
조각 하나가 끝날 때마다 결과를 파일로 남기므로, 중간에 끊겨도 다시 실행하면 끝난 조각은 건너뛴다.

실행:  python double_stage1.py                 (하이브리드 모델, 5조각)
       python double_stage1.py --arch base     (model.py 모델)

결과물:
- checkpoints/double_stage1_fold{k}.pt     조각 k를 빼고 학습한 1단계 모델
- outputs_double/stage1_oof_fold{k}.csv    조각 k의 예측문 (ID, input, pred, output)
- outputs_double/stage1_oof.csv            train 전체의 예측문 (2단계 학습 데이터)
- outputs_double/stage1_test.csv           test 예측문 (ID, input, pred) (2단계 추론 입력)
- outputs_double/stage1_submission.csv     1단계 앙상블만으로 만든 제출 파일 (2단계와 비교용)
"""
import argparse
import math
import os
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from augmented_train import ChunkDataset, augment_rows, char_f1, evaluate, model_inputs, set_seed, to_device
from data_preprocessing import (
    IGNORE_INDEX, ObfuscationDataset, TokenBudgetSampler, Vocab, clean, collate_fn, read_csv, split_rows, write_csv,
)
from model import DeobfuscationModel
from pretrain_model import MODEL_NAME, PretrainDeobfuscationModel
from pretrain_train import param_groups

OOF_FIELDS = ['ID', 'input', 'pred', 'output']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_path', default='data/train.csv')
    p.add_argument('--test_path', default='data/test.csv')
    p.add_argument('--ckpt_dir', default='checkpoints')
    p.add_argument('--output_dir', default='outputs_double')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--valid_ratio', type=float, default=0.03,
                   help='조각별 학습 데이터에서 조기 종료용으로 떼어 두는 비율 (예측할 조각과는 별개)')
    p.add_argument('--seed', type=int, default=42)
    # 증강 (augmented_train.py와 동일)
    p.add_argument('--aug_times', type=int, default=1, help='학습셋에 증강을 적용해 추가하는 횟수 (0이면 증강 없음)')
    p.add_argument('--aug_ratio', type=float, default=0.5, help='1회당 부분 복원 증강본을 만드는 문장 비율')
    p.add_argument('--dict_ratio', type=float, default=0.5, help='1회당 단어 사전 치환 증강본을 만드는 문장 비율')
    p.add_argument('--threshold', type=float, default=0.2, help='사전 치환 증강본에서 단어가 치환될 확률')
    p.add_argument('--obf_ratio', type=float, default=0.5, help='1회당 재난독화 증강본을 만드는 문장 비율 (0이면 끔)')
    p.add_argument('--concat_ratio', type=float, default=0.1,
                   help='문장을 이어 붙인 긴 샘플의 수 (증강까지 끝난 학습 문장 수 대비 비율, 0이면 끔)')
    p.add_argument('--concat_min_len', type=int, default=400, help='이어 붙인 샘플의 최소 목표 길이')
    p.add_argument('--concat_max_len', type=int, default=1600, help='이어 붙인 샘플의 최대 목표 길이')
    # 모델
    p.add_argument('--arch', choices=['hybrid', 'base'], default='hybrid',
                   help='hybrid: pretrain_model.py (model.py 구조 + ELECTRA 상위 층), base: model.py')
    p.add_argument('--model_name', default=MODEL_NAME, help='hybrid에서 상위 층을 가져올 사전학습 모델')
    p.add_argument('--d_model', type=int, default=256, help='base 전용 (hybrid는 사전학습 모델의 hidden size로 고정)')
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--n_layers', type=int, default=6)
    p.add_argument('--n_pretrained_layers', type=int, default=6, help='hybrid 전용: 위에 얹는 사전학습 상위 층 수')
    p.add_argument('--d_ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.15)
    # 학습
    p.add_argument('--max_tokens', type=int, default=16384, help='배치당 (문장 수 x 최대 길이) 상한, GPU 메모리에 맞게 조절')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--lr', type=float, default=3e-4, help='처음부터 학습하는 층의 학습률')
    p.add_argument('--pretrained_lr', type=float, default=1e-4, help='hybrid 전용: 사전학습 층 중 최상위 층의 학습률')
    p.add_argument('--layer_decay', type=float, default=0.9, help='hybrid 전용: 사전학습 층에서 아래 층으로 갈수록 학습률에 곱하는 비율')
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.05)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--ema_decay', type=float, default=0.999, help='가중치 EMA의 감쇠율 (0이면 EMA 없이 학습 중인 가중치 그대로)')
    p.add_argument('--patience', type=int, default=5, help='검증 F1이 이 epoch 수 동안 오르지 않으면 조기 종료')
    p.add_argument('--infer_window', type=int, default=0,
                   help='예측할 때 이보다 긴 문장은 이 길이의 겹치는 구간으로 나눠 넣는다 (0이면 문장 전체를 한 번에)')
    p.add_argument('--infer_overlap', type=int, default=64, help='구간으로 나눌 때 이웃 구간이 겹치는 문자 수')
    p.add_argument('--no_amp', action='store_true', help='fp16 혼합 정밀도 끄기')
    return p.parse_args()


def build_model(args, vocab):
    """새 모델과, 체크포인트만으로 같은 구조를 복원할 수 있는 설정을 만든다."""
    config = dict(
        n_chars=len(vocab.chars), n_syllables=len(vocab.syllables),
        n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    )
    if args.arch == 'hybrid':
        model = PretrainDeobfuscationModel(
            **config, n_pretrained_layers=args.n_pretrained_layers, model_name=args.model_name,
        )
        config['electra_config'] = model.electra_config.to_dict()
    else:
        config['d_model'] = args.d_model
        model = DeobfuscationModel(**config)
    return model, config


def load_model(ckpt, device):
    model_class = PretrainDeobfuscationModel if ckpt['arch'] == 'hybrid' else DeobfuscationModel
    model = model_class(**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model'])
    return model


def build_optimizer(model, args):
    if args.arch == 'hybrid':
        return torch.optim.AdamW(param_groups(model, args), lr=args.lr, betas=(0.9, 0.999), eps=1e-6)
    # LayerNorm, bias, 임베딩은 weight decay 제외
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and 'embedding' not in n]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2 or 'embedding' in n]
    return torch.optim.AdamW(
        [{'params': decay, 'weight_decay': args.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}],
        lr=args.lr, betas=(0.9, 0.98),
    )


def train_fold(train_rows, valid_rows, vocab, args, device, ckpt_path):
    """train_rows로 학습하고 valid_rows의 F1이 가장 높은 시점의 EMA 가중치를 ckpt_path에 저장한다."""
    train_set = ObfuscationDataset(train_rows, vocab)
    valid_set = ObfuscationDataset(valid_rows, vocab)
    train_loader = DataLoader(
        train_set, batch_sampler=TokenBudgetSampler(train_set.lengths, args.max_tokens, shuffle=True, seed=args.seed),
        collate_fn=collate_fn, pin_memory=device.type == 'cuda',
    )
    model, model_config = build_model(args, vocab)
    model.to(device)
    print(f'파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(args.ema_decay),
    )
    optimizer = build_optimizer(model, args)
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

        valid_f1, _, _ = evaluate(ema.module, valid_set, vocab, args, device)
        improved = valid_f1 > best_f1
        print(
            f'epoch {epoch:3d} | loss {total_loss / max(total_tokens, 1):.4f} | valid char F1 {valid_f1:.4f}'
            f' | lr {scheduler.get_last_lr()[0]:.2e} | {time.time() - start:.0f}s' + (' *' if improved else '')
        )
        if improved:
            best_f1, bad_epochs = valid_f1, 0
            torch.save(
                {'model': ema.module.state_dict(), 'model_config': model_config, 'arch': args.arch,
                 'vocab': vocab.state_dict(), 'valid_f1': valid_f1, 'epoch': epoch},
                ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f'조기 종료: {args.patience} epoch 동안 개선 없음')
                break
    print(f'best valid char F1: {best_f1:.4f}')


@torch.no_grad()
def ensemble_predict(models, dataset, vocab, args, device):
    """모델들의 확률을 평균해 예측한 문자열 리스트 (모델이 하나면 그 모델의 예측).

    구간 처리는 augmented_train.predict와 같다. 모든 모델이 같은 vocab으로 학습되어 있어야 한다.
    """
    for model in models:
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
        probs = 0
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda' and not args.no_amp):
            for model in models:
                probs = probs + model(**model_inputs(batch, target_mask)).float().softmax(-1)
        pred_ids = probs.argmax(-1).tolist()
        offset = 0
        for idx, n in zip(batch['idx'], target_mask.sum(1).tolist()):
            chunk_preds[idx] = pred_ids[offset:offset + n]
            offset += n
    sent_preds = [[] for _ in dataset.inputs]
    for (sent, *_), ids in zip(chunked.chunks, chunk_preds):
        sent_preds[sent] += ids
    return [vocab.decode(text, ids) for text, ids in zip(dataset.inputs, sent_preds)]


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    rows = read_csv(args.train_path)
    # 사전은 train 전체로 한 번만 만든다. 조각별 모델의 출력 클래스가 같아야 확률을 평균할 수 있다
    # (예측할 조각의 정답 문장을 학습하는 것이 아니라 음절 목록만 공유하는 것)
    vocab = Vocab.build(rows)
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    fold_of = {rows[i]['ID']: rank % args.n_folds for rank, i in enumerate(order)}

    ckpt_paths = [os.path.join(args.ckpt_dir, f'double_stage1_fold{k}.pt') for k in range(args.n_folds)]
    oof_paths = [os.path.join(args.output_dir, f'stage1_oof_fold{k}.csv') for k in range(args.n_folds)]
    for k in range(args.n_folds):
        if os.path.exists(oof_paths[k]) and os.path.exists(ckpt_paths[k]):
            print(f'[조각 {k}] 이미 끝남 -> 건너뜀 ({oof_paths[k]})')
            continue
        held_rows = [row for row in rows if fold_of[row['ID']] == k]
        fold_rows = [row for row in rows if fold_of[row['ID']] != k]
        # 조기 종료용 검증셋은 학습 쪽에서 따로 뗀다. 예측할 조각으로 체크포인트를 고르면 그 조각의 예측이 실제보다 좋아진다
        train_rows, valid_rows = split_rows(fold_rows, args.valid_ratio, args.seed + k)
        n_original = len(train_rows)
        train_rows = augment_rows(train_rows, args)
        print(
            f'[조각 {k}] train {len(train_rows)} (원본 {n_original} + 증강 {len(train_rows) - n_original})'
            f' / valid {len(valid_rows)} / 예측할 조각 {len(held_rows)}'
        )
        train_fold(train_rows, valid_rows, vocab, args, device, ckpt_paths[k])

        model = load_model(torch.load(ckpt_paths[k], map_location=device), device)
        held_set = ObfuscationDataset(held_rows, vocab)
        preds = ensemble_predict([model], held_set, vocab, args, device)
        f1s = [char_f1(p, t) for p, t in zip(preds, held_set.outputs)]
        write_csv(
            oof_paths[k],
            [{'ID': i, 'input': x, 'pred': p, 'output': y}
             for i, x, p, y in zip(held_set.ids, held_set.inputs, preds, held_set.outputs)],
            OOF_FIELDS,
        )
        print(f'[조각 {k}] 안 본 조각 char F1 {sum(f1s) / len(f1s):.4f} -> {oof_paths[k]}')
        del model

    # train 전체의 예측문을 원래 순서로 합친다
    oof = {row['ID']: row for path in oof_paths for row in read_csv(path)}
    oof_rows = [oof[row['ID']] for row in rows]
    oof_path = os.path.join(args.output_dir, 'stage1_oof.csv')
    write_csv(oof_path, oof_rows, OOF_FIELDS)
    f1s = [char_f1(row['pred'], clean(row['output'])) for row in oof_rows]
    print(f'train 전체(안 본 조각 예측) char F1 {sum(f1s) / len(f1s):.4f} -> {oof_path}')

    # test: 조각별 모델의 확률을 평균
    models = [load_model(torch.load(path, map_location=device), device) for path in ckpt_paths]
    test_set = ObfuscationDataset(read_csv(args.test_path), vocab, with_label=False)
    test_preds = ensemble_predict(models, test_set, vocab, args, device)
    test_path = os.path.join(args.output_dir, 'stage1_test.csv')
    write_csv(
        test_path,
        [{'ID': i, 'input': x, 'pred': p} for i, x, p in zip(test_set.ids, test_set.inputs, test_preds)],
        ['ID', 'input', 'pred'],
    )
    submission_path = os.path.join(args.output_dir, 'stage1_submission.csv')
    write_csv(submission_path, [{'ID': i, 'output': p} for i, p in zip(test_set.ids, test_preds)], ['ID', 'output'])
    print(f'test 예측 {len(test_preds)}개 ({len(models)}개 모델 평균) -> {test_path}, {submission_path}')


if __name__ == '__main__':
    main()
