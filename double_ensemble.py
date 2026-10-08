"""2단계 교정의 반복 적용 + 앙상블 (학습 없이 예측만 수행). 최종 제출 파일을 만든다.

- 반복 적용: 교정한 출력을 다시 1단계 예측문 자리에 넣어 n_passes번까지 교정한다.
  첫 회에 주변 글자가 고쳐지면 다음 회에 문맥이 좋아져 남은 글자를 더 고칠 수 있다.
  검증 F1이 가장 높은 횟수를 골라 test에도 같은 횟수를 적용한다.
- 앙상블: 2단계 체크포인트를 여러 개 주면 확률을 글자마다 (가중) 평균한다.
  모두 같은 검증 문장과 같은 학습 데이터로 학습한 것이어야 한다
  (double_train.py에서 --seed는 같게 두고 --model_seed만 바꿔 학습한다).
- copy_margin: 지정하지 않으면 margin_sweep 중 검증 F1(1회 적용 기준)이 가장 높은 값을 쓴다.

실행:  python double_ensemble.py --ckpt_paths checkpoints/double_stage2.pt
       python double_ensemble.py --ckpt_paths checkpoints/double_stage2.pt checkpoints/double_stage2_s1.pt

결과물:
- outputs_double_final/val_predictions.csv   검증셋 ID, 1단계 F1, 최종 F1, input, 1단계 예측, 정답, 최종 예측 (F1 낮은 순)
- outputs_double_final/submission.csv        test 예측 (sample_submission 형식)
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from data_preprocessing import TokenBudgetSampler, clean, is_hangul, read_csv, write_csv
from double_data import Stage2Dataset, collate_fn, model_inputs, target_positions
from double_train import count_changes, load_checkpoint
from training import char_f1, mean_char_f1, to_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_paths', nargs='+', required=True, help='2단계 체크포인트들 (공백으로 구분)')
    p.add_argument('--weights', type=float, nargs='+', default=None,
                   help='체크포인트별 가중치 (ckpt_paths와 같은 순서, 같은 개수). 지정하지 않으면 모두 1')
    p.add_argument('--oof_path', default='outputs_double/stage1_oof.csv', help='1단계의 train 예측문 (ID, input, pred, output)')
    p.add_argument('--stage1_test_path', default='outputs_double/stage1_test.csv', help='1단계의 test 예측문 (ID, input, pred)')
    p.add_argument('--output_dir', default='outputs_double_final')
    p.add_argument('--n_passes', type=int, default=3, help='교정을 반복 적용해 볼 최대 횟수 (1이면 한 번만)')
    p.add_argument('--copy_margin', type=float, default=None,
                   help='1단계(또는 앞 회차) 음절의 logit에 더하는 값. 지정하지 않으면 margin_sweep 중 검증 F1이 가장 높은 값')
    p.add_argument('--margin_sweep', default='0,0.5,1,1.5,2,3,4', help='copy_margin 후보 (쉼표로 구분)')
    p.add_argument('--topk', type=int, default=10, help='모델마다 글자별로 남기는 상위 후보 수 (나머지 확률은 무시)')
    p.add_argument('--max_chars', type=int, default=510, help='구간당 최대 문자 수 (학습 때와 같게)')
    p.add_argument('--overlap', type=int, default=128, help='긴 문장을 나눌 때 이웃 구간이 겹치는 문자 수')
    p.add_argument('--max_tokens', type=int, default=32768, help='배치당 (구간 수 x 최대 길이) 상한의 절반 (예측은 이 값의 2배를 쓴다)')
    p.add_argument('--no_amp', action='store_true', help='fp16 혼합 정밀도 끄기')
    return p.parse_args()


@torch.no_grad()
def member_topk(member, rows, args, device):
    """rows(ID, input, pred)의 문장마다 한글 위치별 상위 후보 (ids, logits)를 반환한다. 둘 다 (한글 수, topk).

    member는 load_checkpoint가 돌려주는 (모델, 사전, 토크나이저, 체크포인트)다.
    """
    model, vocab, tokenizer, _ = member
    dataset = Stage2Dataset(rows, vocab, tokenizer, args, with_label=False)
    loader = DataLoader(
        dataset, batch_sampler=TokenBudgetSampler(dataset.lengths, args.max_tokens * 2, shuffle=False),
        collate_fn=collate_fn, pin_memory=device.type == 'cuda',
    )
    chunk_out = [None] * len(dataset)
    for batch in loader:
        batch = to_device(batch, device)
        target_mask = target_positions(batch)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda' and not args.no_amp):
            logits = model(**model_inputs(batch, target_mask))
        values, ids = logits.float().topk(min(args.topk, logits.size(-1)), dim=-1)
        values, ids = values.cpu(), ids.cpu()
        offset = 0
        for idx, n in zip(batch['idx'], target_mask.sum(1).tolist()):
            chunk_out[idx] = (ids[offset:offset + n], values[offset:offset + n])
            offset += n
    # 구간이 문장 순, 위치 순이라 차례로 이어 붙이면 문장 전체가 된다
    sents = [[] for _ in rows]
    for (sent, *_), out in zip(dataset.chunks, chunk_out):
        sents[sent].append(out)
    return [(torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])) for outs in sents]


def combine(rows, member_outs, vocab, margin, weights=None):
    """모델들의 확률을 가중 평균해 문장별 예측 문자열을 만든다. member_outs: 모델별 member_topk 결과."""
    weights = weights or [1.0] * len(member_outs)
    preds = []
    for i, row in enumerate(rows):
        # 지금 예측문(row['pred'])의 음절 id. 이 음절의 logit에 margin을 더해, 다른 음절이 그만큼 더 확실할 때만 바꾼다
        copy = torch.tensor(
            [vocab.syl2id.get(p, -1) for x, p in zip(row['input'], row['pred']) if is_hangul(x)], dtype=torch.long,
        )
        probs = torch.zeros(len(copy), len(vocab.syllables))
        for outs, weight in zip(member_outs, weights):
            ids, logits = outs[i]
            if margin:
                logits = logits + margin * (ids == copy[:, None])
            probs.scatter_add_(1, ids, weight * logits.softmax(-1))
        preds.append(vocab.decode(row['input'], probs.argmax(-1).tolist()))
    return preds


def run_passes(members, rows, vocab, args, device, margin, n_passes):
    """교정을 n_passes번 반복 적용하고 회차별 예측을 리스트로 반환한다. 2회차부터는 앞 회차의 출력이 입력 예측문이 된다."""
    history = []
    for _ in range(n_passes):
        outs = [member_topk(member, rows, args, device) for member in members]
        preds = combine(rows, outs, vocab, margin, args.weights)
        history.append(preds)
        rows = [{'ID': row['ID'], 'input': row['input'], 'pred': pred} for row, pred in zip(rows, preds)]
    return history


def main():
    args = parse_args()
    assert args.weights is None or len(args.weights) == len(args.ckpt_paths), '--weights의 개수가 --ckpt_paths와 다르다'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    members = [load_checkpoint(path, device) for path in args.ckpt_paths]
    for path, (_, _, _, ckpt) in zip(args.ckpt_paths, members):
        print(f"{path}: epoch {ckpt['epoch']}, valid F1 {ckpt['valid_f1']:.4f}")
    if args.weights:
        print('가중치: ' + ', '.join(f'{path} {w:g}' for path, w in zip(args.ckpt_paths, args.weights)))
    vocab = members[0][1]
    valid_ids = set(members[0][3]['valid_ids'])
    for path, (_, other_vocab, _, ckpt) in zip(args.ckpt_paths[1:], members[1:]):
        assert set(ckpt['valid_ids']) == valid_ids, (
            f'{path}: 검증 문장이 {args.ckpt_paths[0]}와 다르다. 같은 --seed, --valid_ratio로 학습한 체크포인트만 섞을 수 있다'
        )
        assert other_vocab.syllables == vocab.syllables, (
            f'{path}: 출력 음절 사전이 {args.ckpt_paths[0]}와 다르다. 같은 학습 데이터로 학습한 체크포인트만 섞을 수 있다'
        )
    os.makedirs(args.output_dir, exist_ok=True)

    valid_rows = [row for row in read_csv(args.oof_path) if row['ID'] in valid_ids]
    outputs = [clean(row['output']) for row in valid_rows]
    stage1 = [row['pred'] for row in valid_rows]
    print(f'valid {len(valid_rows)} | 교정 전 valid char F1 {mean_char_f1(stage1, outputs):.4f}')

    # 1회 적용: 모델별 후보를 한 번만 계산해 두고 margin 선택과 모델별 단독 점수에 함께 쓴다
    outs = [member_topk(member, valid_rows, args, device) for member in members]
    if args.copy_margin is None:
        # 검증 F1이 가장 높은 margin을 고른다 (검증셋으로 고른 값이라 아래 검증 F1은 약간 낙관적이다)
        scores = {
            m: mean_char_f1(combine(valid_rows, outs, vocab, m, args.weights), outputs)
            for m in map(float, args.margin_sweep.split(','))
        }
        margin = max(scores, key=scores.get)
        print('copy_margin별 valid char F1: ' + ', '.join(f'{m:g}: {f:.4f}' for m, f in scores.items()) + f' -> {margin:g} 선택')
    else:
        margin = args.copy_margin
    if len(members) > 1:
        for path, out in zip(args.ckpt_paths, outs):
            print(f'  단독 {path}: {mean_char_f1(combine(valid_rows, [out], vocab, margin), outputs):.4f}')

    # 반복 적용: 회차별 검증 F1을 보고 가장 좋은 횟수를 고른다
    history = run_passes(members, valid_rows, vocab, args, device, margin, args.n_passes)
    f1_by_pass = [mean_char_f1(preds, outputs) for preds in history]
    best_pass = max(range(len(history)), key=lambda i: f1_by_pass[i]) + 1
    print(
        f'모델 {len(members)}개, 적용 횟수별 valid char F1: '
        + ', '.join(f'{i + 1}회: {f:.4f}' for i, f in enumerate(f1_by_pass)) + f' -> {best_pass}회 선택'
    )
    preds = history[best_pass - 1]

    result = [
        {'ID': row['ID'], 'f1_stage1': round(char_f1(s, y), 4), 'f1': round(char_f1(p, y), 4),
         'input': row['input'], 'stage1': s, 'output': y, 'pred': p}
        for row, s, y, p in zip(valid_rows, stage1, outputs, preds)
    ]
    result.sort(key=lambda r: r['f1'])  # 틀린 문장부터 보이도록
    valid_path = os.path.join(args.output_dir, 'val_predictions.csv')
    write_csv(valid_path, result, ['ID', 'f1_stage1', 'f1', 'input', 'stage1', 'output', 'pred'])
    print(f'valid char F1: 교정 전 {mean_char_f1(stage1, outputs):.4f} -> 교정 후 {f1_by_pass[best_pass - 1]:.4f} -> {valid_path}')
    fixed, broken, still_wrong = count_changes([row['input'] for row in valid_rows], stage1, outputs, preds)
    print(f'글자 단위: 고침 {fixed} / 망침 {broken} / 여전히 틀림 {still_wrong}')

    # test 추론 (검증에서 고른 margin과 적용 횟수 그대로)
    test_rows = read_csv(args.stage1_test_path)
    test_preds = run_passes(members, test_rows, vocab, args, device, margin, best_pass)[-1]
    n_changed = sum(p != row['pred'] for p, row in zip(test_preds, test_rows))
    submission_path = os.path.join(args.output_dir, 'submission.csv')
    write_csv(submission_path, [{'ID': row['ID'], 'output': p} for row, p in zip(test_rows, test_preds)], ['ID', 'output'])
    print(f'test 예측 {len(test_preds)}개 (1단계 예측에서 바뀐 문장 {n_changed}개) -> {submission_path}')


if __name__ == '__main__':
    main()
