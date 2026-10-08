"""2단계 교정 모델 학습 / 평가 / 추론.

double_stage1.py가 만든 1단계 예측문을 입력으로, 틀린 글자를 고치는 교정 모델(double_model.py)을 학습한다.
학습 데이터는 stage1_oof.csv(1단계가 처음 보는 문장을 예측한 결과)라, 검증/test에서 나오는 것과 같은 성격의 실수가 담겨 있다.
KoCharELECTRA는 위치 임베딩이 512개라 [CLS]/[SEP]를 뺀 510자를 넘는 문장은 겹치는 구간으로 나눠 넣고,
예측할 때는 각 문자를 문맥이 더 넓게 보이는 구간에서 가져와 다시 이어 붙인다.

먼저:      python double_stage1.py
학습:      python double_train.py
평가만:    python double_train.py --eval_only   (저장된 체크포인트로 검증셋 평가 + test 추론)

결과물:
- checkpoints/double_stage2.pt           검증 F1이 가장 높은 교정 모델
- outputs_double/val_predictions.csv     검증셋 ID, 1단계 F1, 2단계 F1, input, 1단계 예측, 정답, 2단계 예측 (F1 낮은 순)
- outputs_double/submission.csv          test 예측 (sample_submission 형식)
"""
import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from augmented_train import char_f1, set_seed, split_chunks, to_device
from data_preprocessing import (
    IGNORE_INDEX, TokenBudgetSampler, Vocab, clean, is_hangul, read_csv, split_rows, write_csv,
)
from double_model import MODEL_NAME, DoubleCorrectionModel, load_tokens

AUG_SUFFIX = '_obf'  # double_augment.py가 추가 예측문의 ID에 붙이는 접미사 (뒤에 번호가 온다)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--oof_path', default='outputs_double/stage1_oof.csv', help='1단계의 train 예측문 (ID, input, pred, output)')
    p.add_argument('--extra_oof_path', default=None,
                   help='double_augment.py가 만든 추가 예측문. 지정하면 학습셋에만 더한다 (검증 문장에서 나온 것은 제외)')
    p.add_argument('--stage1_test_path', default='outputs_double/stage1_test.csv', help='1단계의 test 예측문 (ID, input, pred)')
    p.add_argument('--ckpt_path', default='checkpoints/double_stage2.pt')
    p.add_argument('--output_dir', default='outputs_double')
    p.add_argument('--valid_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42)
    # 모델
    p.add_argument('--model_name', default=MODEL_NAME)
    p.add_argument('--dropout', type=float, default=0.1, help='분류 헤드 앞 dropout (인코더 내부는 사전학습 설정 유지)')
    p.add_argument('--copy_bias', type=float, default=5.0, help='1단계가 예측한 음절의 logit에 더하는 값의 초기값 (학습됨)')
    p.add_argument('--max_chars', type=int, default=510, help='구간당 최대 문자 수 (512 - [CLS]/[SEP])')
    p.add_argument('--overlap', type=int, default=128, help='긴 문장을 나눌 때 이웃 구간이 겹치는 문자 수')
    # 학습
    p.add_argument('--max_tokens', type=int, default=8192, help='배치당 (구간 수 x 최대 길이) 상한, GPU 메모리에 맞게 조절')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=1e-4, help='인코더 최상위 층의 학습률')
    p.add_argument('--layer_decay', type=float, default=0.9, help='아래 층으로 갈수록 학습률에 곱하는 비율 (1이면 모든 층 동일)')
    p.add_argument('--head_lr', type=float, default=1e-3, help='새로 초기화한 분류 헤드/자모 임베딩/copy_bias의 학습률')
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.1)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--ema_decay', type=float, default=0.99, help='가중치 EMA의 감쇠율 (0이면 EMA 없이 학습 중인 가중치 그대로)')
    p.add_argument('--patience', type=int, default=4, help='검증 F1이 이 epoch 수 동안 오르지 않으면 조기 종료')
    # 추론
    p.add_argument('--copy_margin', type=float, default=None,
                   help='예측할 때 1단계 음절의 logit에 더하는 값. 클수록 확신이 있을 때만 바꾼다. 지정하지 않으면 margin_sweep 중 검증 F1이 가장 높은 값')
    p.add_argument('--margin_sweep', default='0,0.5,1,1.5,2,3,4', help='copy_margin 후보 (쉼표로 구분)')
    p.add_argument('--no_amp', action='store_true', help='fp16 혼합 정밀도 끄기')
    p.add_argument('--eval_only', action='store_true')
    return p.parse_args()


class DoubleDataset(Dataset):
    """1단계 예측문을 KoCharELECTRA 토큰 id로 바꾸고 구간 단위로 꺼내는 데이터셋. 길이/인덱스는 모두 구간 기준.

    rows의 'input'은 난독화 문장, 'pred'는 1단계 예측문, 'output'은 정답 (with_label일 때만 필요).
    """

    def __init__(self, rows, vocab, token2id, args, with_label=True):
        self.ids = [row['ID'] for row in rows]
        self.inputs = [row['input'] for row in rows]
        self.stage1 = [row['pred'] for row in rows]
        self.outputs = [clean(row['output']) for row in rows] if with_label else None
        unk, self.cls, self.sep = token2id['[UNK]'], token2id['[CLS]'], token2id['[SEP]']
        self.pad = token2id['[PAD]']
        self.features, self.labels, self.chunks = [], [], []  # chunks: (문장 idx, start, end, keep_start, keep_end)
        for i, (text, pred) in enumerate(zip(self.inputs, self.stage1)):
            assert len(text) == len(pred), f'{self.ids[i]}: 1단계 예측문의 길이가 input과 다르다'
            _, cho, jung, jong = vocab.encode_input(text)
            input_ids = torch.tensor([token2id.get(ch, unk) for ch in pred], dtype=torch.long)
            # 1단계가 예측한 음절의 출력 사전 id. 한글 위치가 아니거나 사전에 없는 음절이면 -1
            copy_ids = torch.tensor(
                [vocab.syl2id.get(p, -1) if is_hangul(x) else -1 for x, p in zip(text, pred)], dtype=torch.long,
            )
            self.features.append((input_ids, torch.stack((cho, jung, jong), dim=1), copy_ids))
            if with_label:
                self.labels.append(vocab.encode_label(text, self.outputs[i]))
            self.chunks += [(i, *chunk) for chunk in split_chunks(len(text), args.max_chars, args.overlap)]
        self.lengths = [end - start + 2 for _, start, end, _, _ in self.chunks]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        sent, start, end, keep_start, keep_end = self.chunks[idx]
        n = end - start
        ids, sent_jamo, sent_copy = self.features[sent]
        # 앞뒤에 [CLS], [SEP]를 붙이므로 모든 위치가 1칸씩 밀린다
        input_ids = torch.full((n + 2,), self.cls, dtype=torch.long)
        input_ids[1:-1], input_ids[-1] = ids[start:end], self.sep
        jamo = torch.zeros(n + 2, 3, dtype=torch.long)
        jamo[1:-1] = sent_jamo[start:end]
        copy_ids = torch.full((n + 2,), -1, dtype=torch.long)
        copy_ids[1:-1] = sent_copy[start:end]
        keep = torch.zeros(n + 2, dtype=torch.bool)
        keep[1 + keep_start - start:1 + keep_end - start] = True
        labels = torch.full((n + 2,), IGNORE_INDEX, dtype=torch.long)
        if self.labels:
            labels[1:-1] = self.labels[sent][start:end]
        return {'idx': idx, 'input_ids': input_ids, 'jamo': jamo, 'copy_ids': copy_ids, 'keep': keep,
                'labels': labels, 'pad': self.pad}


def collate_fn(batch):
    def pad(key, value):
        return pad_sequence([item[key] for item in batch], batch_first=True, padding_value=value)

    input_ids = pad('input_ids', batch[0]['pad'])
    lengths = torch.tensor([len(item['input_ids']) for item in batch])
    return {
        'idx': [item['idx'] for item in batch],
        'input_ids': input_ids, 'jamo': pad('jamo', 0), 'copy_ids': pad('copy_ids', -1),
        'attention_mask': (torch.arange(input_ids.size(1))[None] < lengths[:, None]).long(),
        'keep': pad('keep', False), 'labels': pad('labels', IGNORE_INDEX),
    }


def model_inputs(batch, target_mask):
    return dict(
        input_ids=batch['input_ids'], jamo=batch['jamo'], attention_mask=batch['attention_mask'],
        target_mask=target_mask, copy_ids=batch['copy_ids'][target_mask],
    )


@torch.no_grad()
def predict(model, dataset, vocab, args, device, margin=0.0):
    """데이터셋 전체를 예측해 문자열 리스트로 반환 (input 순서 유지).

    margin만큼 1단계 음절의 logit을 올려, 다른 음절이 그만큼 더 확실할 때만 바꾼다.
    """
    model.eval()
    loader = DataLoader(
        dataset, batch_sampler=TokenBudgetSampler(dataset.lengths, args.max_tokens * 2, shuffle=False),
        collate_fn=collate_fn, pin_memory=device.type == 'cuda',
    )
    chunk_preds = [None] * len(dataset)
    for batch in loader:
        batch = to_device(batch, device)
        target_mask = (batch['jamo'][..., 0] > 0) & batch['keep']  # 이 구간이 맡은 한글 음절 위치
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda' and not args.no_amp):
            logits = model(**model_inputs(batch, target_mask))
        if margin:
            copy_ids = batch['copy_ids'][target_mask]
            has_copy = (copy_ids >= 0).nonzero(as_tuple=True)[0]
            logits[has_copy, copy_ids[has_copy]] += margin
        pred_ids = logits.argmax(-1).tolist()
        # 배치 전체에서 모은 한글 위치 예측을 구간별로 다시 나눔
        offset = 0
        for idx, n in zip(batch['idx'], target_mask.sum(1).tolist()):
            chunk_preds[idx] = pred_ids[offset:offset + n]
            offset += n
    # dataset.chunks는 문장 순, 문장 안에서는 위치 순이라 차례로 이어 붙이면 문장 전체 예측이 된다
    sent_preds = [[] for _ in dataset.inputs]
    for (sent, *_), ids in zip(dataset.chunks, chunk_preds):
        sent_preds[sent] += ids
    return [vocab.decode(text, ids) for text, ids in zip(dataset.inputs, sent_preds)]


def evaluate(model, dataset, vocab, args, device, margin=0.0):
    preds = predict(model, dataset, vocab, args, device, margin)
    f1s = [char_f1(p, t) for p, t in zip(preds, dataset.outputs)]
    return sum(f1s) / len(f1s), preds, f1s


def stage1_f1(dataset):
    """교정 전(1단계 예측문 그대로)의 문장별 F1."""
    return [char_f1(p, t) for p, t in zip(dataset.stage1, dataset.outputs)]


def param_groups(model, args):
    """새로 초기화한 층은 head_lr, 사전학습 인코더는 아래 층일수록 작은 학습률 (layer-wise lr decay).

    LayerNorm, bias는 weight decay 제외.
    """
    layers = model.electra.encoder.layer
    n_layers = len(layers)
    blocks = [(model.electra.embeddings, args.lr * args.layer_decay ** n_layers)]
    if hasattr(model.electra, 'embeddings_project'):  # embedding_size != hidden_size일 때만 존재
        blocks.append((model.electra.embeddings_project, args.lr * args.layer_decay ** n_layers))
    blocks += [(layer, args.lr * args.layer_decay ** (n_layers - 1 - i)) for i, layer in enumerate(layers)]
    blocks += [(m, args.head_lr) for m in (model.cho, model.jung, model.jong, model.head)]
    groups = [{'params': [model.copy_bias], 'lr': args.head_lr, 'weight_decay': 0.0}]
    for module, lr in blocks:
        params = [p for p in module.parameters() if p.requires_grad]
        for group, weight_decay in (([p for p in params if p.dim() >= 2], args.weight_decay),
                                    ([p for p in params if p.dim() < 2], 0.0)):
            if group:
                groups.append({'params': group, 'lr': lr, 'weight_decay': weight_decay})
    assert sum(len(g['params']) for g in groups) == sum(p.requires_grad for p in model.parameters())
    return groups


def train(args, device):
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

    train_set = DoubleDataset(train_rows, vocab, token2id, args)
    valid_set = DoubleDataset(valid_rows, vocab, token2id, args)
    base_f1s = stage1_f1(valid_set)
    print(
        f'train {len(train_rows)} / valid {len(valid_rows)} | 구간 수 train {len(train_set)} / valid {len(valid_set)}'
        f' | 출력 음절 {len(vocab.syllables)} | 교정 전 valid char F1 {sum(base_f1s) / len(base_f1s):.4f}'
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

        valid_f1, _, _ = evaluate(ema.module, valid_set, vocab, args, device)
        improved = valid_f1 > best_f1
        print(
            f'epoch {epoch:3d} | loss {total_loss / max(total_tokens, 1):.4f} | valid char F1 {valid_f1:.4f}'
            f' | lr {scheduler.get_last_lr()[-1]:.2e} | {time.time() - start:.0f}s' + (' *' if improved else '')
        )
        if improved:
            best_f1, bad_epochs = valid_f1, 0
            # 평가 때 같은 검증셋을 쓰도록 검증 문장의 ID를 함께 저장 (seed/valid_ratio를 다르게 줘도 섞이지 않는다)
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


def evaluate_and_export(args, device):
    ckpt = torch.load(args.ckpt_path, map_location=device)
    vocab = Vocab.from_state_dict(ckpt['vocab'])
    token2id = {t: i for i, t in enumerate(ckpt['tokens'])}
    model = DoubleCorrectionModel(**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model'])
    print(f"체크포인트 로드: epoch {ckpt['epoch']}, valid F1 {ckpt['valid_f1']:.4f}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 검증셋 평가 결과 (학습 때 저장해 둔 검증 문장만 사용)
    valid_ids = set(ckpt['valid_ids'])
    valid_rows = [row for row in read_csv(args.oof_path) if row['ID'] in valid_ids]
    valid_set = DoubleDataset(valid_rows, vocab, token2id, args)
    base_f1s = stage1_f1(valid_set)
    if args.copy_margin is None:
        # 검증 F1이 가장 높은 margin을 고른다 (검증셋으로 고른 값이라 아래 검증 F1은 약간 낙관적이다)
        scores = {m: evaluate(model, valid_set, vocab, args, device, m)[0] for m in map(float, args.margin_sweep.split(','))}
        margin = max(scores, key=scores.get)
        print('copy_margin별 valid char F1: ' + ', '.join(f'{m:g}: {f:.4f}' for m, f in scores.items()) + f' -> {margin:g} 선택')
    else:
        margin = args.copy_margin
    valid_f1, preds, f1s = evaluate(model, valid_set, vocab, args, device, margin)
    result = [
        {'ID': i, 'f1_stage1': round(b, 4), 'f1': round(f, 4), 'input': x, 'stage1': s, 'output': y, 'pred': p}
        for i, b, f, x, s, y, p in zip(
            valid_set.ids, base_f1s, f1s, valid_set.inputs, valid_set.stage1, valid_set.outputs, preds,
        )
    ]
    result.sort(key=lambda r: r['f1'])  # 틀린 문장부터 보이도록
    valid_path = os.path.join(args.output_dir, 'val_predictions.csv')
    write_csv(valid_path, result, ['ID', 'f1_stage1', 'f1', 'input', 'stage1', 'output', 'pred'])
    print(f'valid char F1: 교정 전 {sum(base_f1s) / len(base_f1s):.4f} -> 교정 후 {valid_f1:.4f} -> {valid_path}')

    # 교정이 글자 단위로 무엇을 바꿨는지 (한글 위치만)
    fixed = broken = still_wrong = changed_wrong = 0
    for x, s, y, p in zip(valid_set.inputs, valid_set.stage1, valid_set.outputs, preds):
        for xc, sc, yc, pc in zip(x, s, y, p):
            if not is_hangul(xc):
                continue
            if sc != yc and pc == yc:
                fixed += 1
            elif sc == yc and pc != yc:
                broken += 1
            elif sc != yc and pc != yc:
                still_wrong += 1
                changed_wrong += sc != pc
    print(
        f'글자 단위: 고침 {fixed} / 망침 {broken} / 여전히 틀림 {still_wrong} (그중 다른 오답으로 바꿈 {changed_wrong})'
    )

    # test 추론
    test_set = DoubleDataset(read_csv(args.stage1_test_path), vocab, token2id, args, with_label=False)
    test_preds = predict(model, test_set, vocab, args, device, margin)
    n_changed = sum(p != s for p, s in zip(test_preds, test_set.stage1))
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    write_csv(submission_path, [{'ID': i, 'output': p} for i, p in zip(test_set.ids, test_preds)], ['ID', 'output'])
    print(f'test 예측 {len(test_preds)}개 (1단계 예측에서 바뀐 문장 {n_changed}개) -> {submission_path}')


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
