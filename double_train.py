"""2단계: 교정 모델 학습 / 평가 / 추론.

double_stage1.py가 만든 1단계 예측문을 입력으로, 틀린 글자를 고치는 교정 모델(double_model.py)을 학습한다.
학습 데이터는 stage1_oof.csv(1단계가 처음 보는 문장을 예측한 결과)라, 검증/test에서 나오는 것과 같은 성격의 실수가 담겨 있다.
학습 입력에는 가리기와 오답으로 바꾸기를 적용한다 (double_data.Stage2Dataset).

기본값은 A100 기준이다. 메모리가 작은 GPU에서는 --max_tokens 8192 --lr 1e-4 --ema_decay 0.99 정도로 낮춘다.

먼저:      python double_stage1.py && python double_augment.py
학습:      python double_train.py --extra_oof_path outputs_double/stage1_oof_aug.csv
평가만:    python double_train.py --eval_only
앙상블용:  python double_train.py --model_seed 1 --ckpt_path checkpoints/double_stage2_s1.pt --output_dir outputs_double_s1 ...
           (--seed는 검증 분할을 정하므로 그대로 두고, 모델 쪽 난수만 --model_seed로 바꾼다)

결과물:
- checkpoints/double_stage2.pt             검증 F1이 가장 높은 교정 모델
- outputs_double/val_predictions.csv       검증셋 ID, 1단계 F1, 2단계 F1, input, 1단계 예측, 정답, 2단계 예측 (F1 낮은 순)
- outputs_double/submission.csv            test 예측 (sample_submission 형식)

교정을 반복 적용하거나 여러 모델을 앙상블한 제출 파일은 double_ensemble.py로 만든다.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data_preprocessing import TokenBudgetSampler, Vocab, is_hangul, read_csv, split_rows, write_csv
from double_data import AUG_SUFFIX, Stage2Dataset, collate_fn, model_inputs, target_positions
from double_model import MODEL_NAME, Stage2Model
from training import char_f1, fit, mean_char_f1, set_seed, to_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--oof_path', default='outputs_double/stage1_oof.csv', help='1단계의 train 예측문 (ID, input, pred, output)')
    p.add_argument('--extra_oof_path', default=None,
                   help='double_augment.py가 만든 추가 예측문. 지정하면 학습셋에만 더한다 (검증 문장에서 나온 것은 제외)')
    p.add_argument('--stage1_test_path', default='outputs_double/stage1_test.csv', help='1단계의 test 예측문 (ID, input, pred)')
    p.add_argument('--ckpt_path', default='checkpoints/double_stage2.pt')
    p.add_argument('--output_dir', default='outputs_double')
    p.add_argument('--valid_ratio', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=42, help='검증 분할과 배치 순서를 정한다. 앙상블할 모델들은 같은 값을 써야 한다')
    p.add_argument('--model_seed', type=int, default=None,
                   help='모델 초기화, dropout, 가리는 위치에 쓰는 난수 seed. 지정하지 않으면 --seed와 같다')
    # 모델
    p.add_argument('--model_name', default=MODEL_NAME)
    p.add_argument('--d_model', type=int, default=384, help='인코더 위에 얹는 글자 단위 층의 차원')
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--n_layers', type=int, default=2, help='글자 단위 층 수')
    p.add_argument('--d_ff', type=int, default=1536)
    p.add_argument('--dropout', type=float, default=0.1, help='글자 단위 층의 dropout (인코더 내부는 사전학습 설정 유지)')
    p.add_argument('--copy_bias', type=float, default=5.0, help='1단계가 예측한 음절의 logit에 더하는 값의 초기값 (학습됨)')
    p.add_argument('--max_chars', type=int, default=510, help='구간당 최대 문자 수 (토큰 수는 문자 수를 넘지 않는다)')
    p.add_argument('--overlap', type=int, default=128, help='긴 문장을 나눌 때 이웃 구간이 겹치는 문자 수')
    # 학습 입력 변형
    p.add_argument('--mask_ratio', type=float, default=0.05, help='학습 때 1단계 예측 글자를 가리는 한글 위치의 비율')
    p.add_argument('--corrupt_ratio', type=float, default=0.03,
                   help='학습 때 1단계 예측 글자를 1단계가 실제로 냈던 오답으로 바꾸는 한글 위치의 비율')
    # 학습
    p.add_argument('--max_tokens', type=int, default=32768, help='배치당 (구간 수 x 최대 문자 수) 상한, GPU 메모리에 맞게 조절')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--lr', type=float, default=2e-4, help='인코더 최상위 층의 학습률')
    p.add_argument('--layer_decay', type=float, default=0.9, help='아래 층으로 갈수록 학습률에 곱하는 비율 (1이면 모든 층 동일)')
    p.add_argument('--head_lr', type=float, default=1e-3, help='새로 초기화한 층(글자 단위 층, 임베딩, 헤드, copy_bias)의 학습률')
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--warmup_ratio', type=float, default=0.1)
    p.add_argument('--label_smoothing', type=float, default=0.1)
    p.add_argument('--ema_decay', type=float, default=0.96, help='가중치 EMA의 감쇠율 (0이면 EMA 없이 학습 중인 가중치 그대로)')
    p.add_argument('--patience', type=int, default=4, help='검증 F1이 이 epoch 수 동안 오르지 않으면 조기 종료')
    # 추론
    p.add_argument('--copy_margin', type=float, default=None,
                   help='예측할 때 1단계 음절의 logit에 더하는 값. 클수록 확신이 있을 때만 바꾼다. 지정하지 않으면 margin_sweep 중 검증 F1이 가장 높은 값')
    p.add_argument('--margin_sweep', default='0,0.5,1,1.5,2,3,4', help='copy_margin 후보 (쉼표로 구분)')
    p.add_argument('--no_amp', action='store_true', help='fp16 혼합 정밀도 끄기')
    p.add_argument('--eval_only', action='store_true')
    return p.parse_args()


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
        target_mask = target_positions(batch)
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


def count_changes(inputs, stage1, outputs, preds):
    """교정이 한글 위치에서 한 일을 센다: (고침, 망침, 여전히 틀림)."""
    fixed = broken = still_wrong = 0
    for x, s, y, p in zip(inputs, stage1, outputs, preds):
        for xc, sc, yc, pc in zip(x, s, y, p):
            if not is_hangul(xc):
                continue
            fixed += sc != yc and pc == yc
            broken += sc == yc and pc != yc
            still_wrong += sc != yc and pc != yc
    return fixed, broken, still_wrong


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
    blocks += [(m, args.head_lr) for m in (
        model.proj, model.obf_embedding, model.pred_embedding, model.layers, model.final_norm, model.head,
    )]
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    train_set = Stage2Dataset(train_rows, vocab, tokenizer, args, train=True)
    valid_set = Stage2Dataset(valid_rows, vocab, tokenizer, args)
    base_f1 = mean_char_f1(valid_set.stage1, valid_set.outputs)
    print(
        f'train {len(train_rows)} / valid {len(valid_rows)} | 구간 수 train {len(train_set)} / valid {len(valid_set)}'
        f' | 출력 음절 {len(vocab.syllables)} | 교정 전 valid char F1 {base_f1:.4f}'
    )
    print(
        f'가리기 {args.mask_ratio:g} / 오답으로 바꾸기 {args.corrupt_ratio:g}'
        f' (1단계가 틀린 적 있는 음절 {len(train_set.confusions)}종)'
    )
    train_loader = DataLoader(
        train_set, batch_sampler=TokenBudgetSampler(train_set.lengths, args.max_tokens, shuffle=True, seed=args.seed),
        collate_fn=collate_fn, pin_memory=device.type == 'cuda',
    )

    model_config = dict(
        n_chars=len(vocab.chars), n_syllables=len(vocab.syllables), d_model=args.d_model,
        n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff, dropout=args.dropout,
    )
    model = Stage2Model(**model_config, model_name=args.model_name, copy_bias=args.copy_bias).to(device)
    # 체크포인트만으로 모델을 복원할 수 있도록 인코더 설정을 함께 저장
    model_config['electra_config'] = model.electra.config.to_dict()
    print(f'파라미터 수: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    optimizer = torch.optim.AdamW(param_groups(model, args), lr=args.lr, betas=(0.9, 0.999), eps=1e-6)

    def evaluate(ema_model):
        # 검증은 가리지 않은 입력으로 한다 (학습 loss는 가린 입력 기준)
        return mean_char_f1(predict(ema_model, valid_set, vocab, args, device), valid_set.outputs)

    def save(ema_model, valid_f1, epoch):
        # 평가 때 같은 검증셋과 토크나이저를 쓰도록 검증 문장의 ID와 모델 이름을 함께 저장
        torch.save(
            {'model': ema_model.state_dict(), 'model_config': model_config, 'vocab': vocab.state_dict(),
             'model_name': args.model_name, 'valid_ids': valid_set.ids, 'valid_f1': valid_f1, 'epoch': epoch},
            args.ckpt_path,
        )

    os.makedirs(os.path.dirname(args.ckpt_path) or '.', exist_ok=True)
    best_f1 = fit(model, train_loader, optimizer, args, device, model_inputs, evaluate, save)
    print(f'best valid char F1: {best_f1:.4f} (교정 전 {base_f1:.4f})')


def load_checkpoint(path, device):
    """체크포인트에서 (모델, 사전, 토크나이저, 체크포인트 dict)를 복원한다."""
    ckpt = torch.load(path, map_location=device)
    model = Stage2Model(**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model'])
    return model.eval(), Vocab.from_state_dict(ckpt['vocab']), AutoTokenizer.from_pretrained(ckpt['model_name']), ckpt


def evaluate_and_export(args, device):
    model, vocab, tokenizer, ckpt = load_checkpoint(args.ckpt_path, device)
    print(f"체크포인트 로드: epoch {ckpt['epoch']}, valid F1 {ckpt['valid_f1']:.4f}")
    os.makedirs(args.output_dir, exist_ok=True)

    # 검증셋 평가 결과 (학습 때 저장해 둔 검증 문장만 사용)
    valid_ids = set(ckpt['valid_ids'])
    valid_rows = [row for row in read_csv(args.oof_path) if row['ID'] in valid_ids]
    valid_set = Stage2Dataset(valid_rows, vocab, tokenizer, args)
    if args.copy_margin is None:
        # 검증 F1이 가장 높은 margin을 고른다 (검증셋으로 고른 값이라 아래 검증 F1은 약간 낙관적이다)
        scores = {
            m: mean_char_f1(predict(model, valid_set, vocab, args, device, m), valid_set.outputs)
            for m in map(float, args.margin_sweep.split(','))
        }
        margin = max(scores, key=scores.get)
        print('copy_margin별 valid char F1: ' + ', '.join(f'{m:g}: {f:.4f}' for m, f in scores.items()) + f' -> {margin:g} 선택')
    else:
        margin = args.copy_margin
    preds = predict(model, valid_set, vocab, args, device, margin)
    result = [
        {'ID': i, 'f1_stage1': round(char_f1(s, y), 4), 'f1': round(char_f1(p, y), 4),
         'input': x, 'stage1': s, 'output': y, 'pred': p}
        for i, x, s, y, p in zip(valid_set.ids, valid_set.inputs, valid_set.stage1, valid_set.outputs, preds)
    ]
    result.sort(key=lambda r: r['f1'])  # 틀린 문장부터 보이도록
    valid_path = os.path.join(args.output_dir, 'val_predictions.csv')
    write_csv(valid_path, result, ['ID', 'f1_stage1', 'f1', 'input', 'stage1', 'output', 'pred'])
    print(
        f'valid char F1: 교정 전 {mean_char_f1(valid_set.stage1, valid_set.outputs):.4f}'
        f' -> 교정 후 {mean_char_f1(preds, valid_set.outputs):.4f} -> {valid_path}'
    )
    fixed, broken, still_wrong = count_changes(valid_set.inputs, valid_set.stage1, valid_set.outputs, preds)
    print(f'글자 단위: 고침 {fixed} / 망침 {broken} / 여전히 틀림 {still_wrong}')

    # test 추론
    test_set = Stage2Dataset(read_csv(args.stage1_test_path), vocab, tokenizer, args, with_label=False)
    test_preds = predict(model, test_set, vocab, args, device, margin)
    n_changed = sum(p != s for p, s in zip(test_preds, test_set.stage1))
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    write_csv(submission_path, [{'ID': i, 'output': p} for i, p in zip(test_set.ids, test_preds)], ['ID', 'output'])
    print(f'test 예측 {len(test_preds)}개 (1단계 예측에서 바뀐 문장 {n_changed}개) -> {submission_path}')


def main():
    args = parse_args()
    # 검증 분할(split_rows)과 배치 순서는 --seed를 따로 받아 쓰므로, 여기서 정하는 것은 모델 쪽 난수뿐이다
    set_seed(args.seed if args.model_seed is None else args.model_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    if not args.eval_only:
        train(args, device)
    evaluate_and_export(args, device)


if __name__ == '__main__':
    main()
