"""2단계 교정 모델 앙상블 + 반복 적용 (학습 없이 예측만 수행).

- 앙상블: 여러 2단계 체크포인트의 확률을 글자마다 평균한다.
  double_train.py / double_mask_train.py (KoCharELECTRA)와 double_kc_train.py (KcELECTRA)의 체크포인트를 섞어도 된다.
  단, 모두 같은 검증 문장과 같은 학습 데이터로 학습한 것이어야 한다 (double_seed_train.py로 만들면 그렇게 된다).
- 반복 적용: 앙상블의 출력을 다시 1단계 예측문 자리에 넣어 n_passes번까지 교정한다.
  검증 F1이 가장 높은 횟수를 골라 test에도 같은 횟수를 적용한다.
- copy_margin: 지정하지 않으면 margin_sweep 중 검증 F1(1회 적용 기준)이 가장 높은 값을 쓴다.

체크포인트가 하나여도 된다 (그러면 '반복 적용'만 확인하는 셈이다).

실행:  python double_ensemble.py --ckpt_paths checkpoints/double_stage2_mask.pt checkpoints/double_stage2_mask_s1.pt

결과물:
- outputs_double_ens/val_predictions.csv   검증셋 ID, 1단계 F1, 최종 F1, input, 1단계 예측, 정답, 최종 예측 (F1 낮은 순)
- outputs_double_ens/submission.csv        test 예측 (sample_submission 형식)
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import double_kc_train
import double_train
from augmented_train import char_f1, to_device
from data_preprocessing import TokenBudgetSampler, Vocab, clean, is_hangul, read_csv, write_csv
from double_kc_model import DoubleKcModel
from double_model import DoubleCorrectionModel


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_paths', nargs='+', required=True, help='2단계 체크포인트들 (공백으로 구분)')
    p.add_argument('--weights', type=float, nargs='+', default=None,
                   help='체크포인트별 가중치 (ckpt_paths와 같은 순서, 같은 개수). 지정하지 않으면 모두 1')
    p.add_argument('--oof_path', default='outputs_double/stage1_oof.csv', help='1단계의 train 예측문 (ID, input, pred, output)')
    p.add_argument('--stage1_test_path', default='outputs_double/stage1_test.csv', help='1단계의 test 예측문 (ID, input, pred)')
    p.add_argument('--output_dir', default='outputs_double_ens')
    p.add_argument('--n_passes', type=int, default=2, help='교정을 반복 적용해 볼 최대 횟수 (1이면 한 번만)')
    p.add_argument('--copy_margin', type=float, default=None,
                   help='1단계(또는 앞 회차) 음절의 logit에 더하는 값. 지정하지 않으면 margin_sweep 중 검증 F1이 가장 높은 값')
    p.add_argument('--margin_sweep', default='0,0.5,1,1.5,2,3,4', help='copy_margin 후보 (쉼표로 구분)')
    p.add_argument('--topk', type=int, default=10, help='모델마다 글자별로 남기는 상위 후보 수 (나머지 확률은 무시)')
    p.add_argument('--max_chars', type=int, default=510, help='구간당 최대 문자 수 (학습 때와 같게)')
    p.add_argument('--overlap', type=int, default=128, help='긴 문장을 나눌 때 이웃 구간이 겹치는 문자 수')
    p.add_argument('--max_tokens', type=int, default=32768, help='배치당 (구간 수 x 최대 길이) 상한의 절반 (예측은 이 값의 2배를 쓴다)')
    p.add_argument('--no_amp', action='store_true', help='fp16 혼합 정밀도 끄기')
    return p.parse_args()


class Member:
    """체크포인트 하나. 종류(KoCharELECTRA / KcELECTRA)에 맞는 데이터셋과 입력 구성을 묶어 둔다."""

    def __init__(self, path, device):
        ckpt = torch.load(path, map_location=device)
        self.path = path
        self.vocab = Vocab.from_state_dict(ckpt['vocab'])
        self.valid_ids = set(ckpt['valid_ids'])
        vocab = self.vocab
        if 'tokens' in ckpt:  # double_train.py / double_mask_train.py
            token2id = {t: i for i, t in enumerate(ckpt['tokens'])}
            self.model = DoubleCorrectionModel(**ckpt['model_config'])
            self.make_dataset = lambda rows, args: double_train.DoubleDataset(rows, vocab, token2id, args, with_label=False)
            self.collate_fn, self.model_inputs = double_train.collate_fn, double_train.model_inputs
            self.target_mask = lambda batch: (batch['jamo'][..., 0] > 0) & batch['keep']
        else:  # double_kc_train.py
            tokenizer = AutoTokenizer.from_pretrained(ckpt['model_name'])
            self.model = DoubleKcModel(**ckpt['model_config'])
            self.make_dataset = lambda rows, args: double_kc_train.DoubleKcDataset(
                rows, vocab, tokenizer, args, with_label=False,
            )
            self.collate_fn, self.model_inputs = double_kc_train.collate_fn, double_kc_train.model_inputs
            self.target_mask = lambda batch: (batch['obf'][..., 1] > 0) & batch['keep']
        self.model.load_state_dict(ckpt['model'])
        self.model.to(device).eval()
        print(f"{path}: epoch {ckpt['epoch']}, valid F1 {ckpt['valid_f1']:.4f}")


@torch.no_grad()
def member_topk(member, rows, args, device):
    """rows(ID, input, pred)의 문장마다 한글 위치별 상위 후보 (ids, logits)를 반환한다. 둘 다 (한글 수, topk)."""
    dataset = member.make_dataset(rows, args)
    loader = DataLoader(
        dataset, batch_sampler=TokenBudgetSampler(dataset.lengths, args.max_tokens * 2, shuffle=False),
        collate_fn=member.collate_fn, pin_memory=device.type == 'cuda',
    )
    chunk_out = [None] * len(dataset)
    for batch in loader:
        batch = to_device(batch, device)
        target_mask = member.target_mask(batch)  # 이 구간이 맡은 한글 음절 위치
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda' and not args.no_amp):
            logits = member.model(**member.model_inputs(batch, target_mask))
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


def mean_f1(preds, outputs):
    return sum(char_f1(p, t) for p, t in zip(preds, outputs)) / len(outputs)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    assert args.weights is None or len(args.weights) == len(args.ckpt_paths), '--weights의 개수가 --ckpt_paths와 다르다'
    members = [Member(path, device) for path in args.ckpt_paths]
    if args.weights:
        print('가중치: ' + ', '.join(f'{path} {w:g}' for path, w in zip(args.ckpt_paths, args.weights)))
    vocab = members[0].vocab
    for member in members[1:]:
        assert member.valid_ids == members[0].valid_ids, (
            f'{member.path}: 검증 문장이 {members[0].path}와 다르다. 같은 --seed, --valid_ratio로 학습한 체크포인트만 섞을 수 있다'
        )
        assert member.vocab.syllables == vocab.syllables, (
            f'{member.path}: 출력 음절 사전이 {members[0].path}와 다르다. 같은 학습 데이터로 학습한 체크포인트만 섞을 수 있다'
        )
    os.makedirs(args.output_dir, exist_ok=True)

    valid_rows = [row for row in read_csv(args.oof_path) if row['ID'] in members[0].valid_ids]
    outputs = [clean(row['output']) for row in valid_rows]
    stage1 = [row['pred'] for row in valid_rows]
    print(f'valid {len(valid_rows)} | 교정 전 valid char F1 {mean_f1(stage1, outputs):.4f}')

    # 1회 적용: 모델별 후보를 한 번만 계산해 두고 margin 선택, 모델별 단독 점수, 앙상블 점수에 함께 쓴다
    outs = [member_topk(member, valid_rows, args, device) for member in members]
    if args.copy_margin is None:
        # 검증 F1이 가장 높은 margin을 고른다 (검증셋으로 고른 값이라 아래 검증 F1은 약간 낙관적이다)
        scores = {
            m: mean_f1(combine(valid_rows, outs, vocab, m, args.weights), outputs)
            for m in map(float, args.margin_sweep.split(','))
        }
        margin = max(scores, key=scores.get)
        print('copy_margin별 valid char F1: ' + ', '.join(f'{m:g}: {f:.4f}' for m, f in scores.items()) + f' -> {margin:g} 선택')
    else:
        margin = args.copy_margin
    if len(members) > 1:
        for member, out in zip(members, outs):
            print(f'  단독 {member.path}: {mean_f1(combine(valid_rows, [out], vocab, margin), outputs):.4f}')

    # 반복 적용: 회차별 검증 F1을 보고 가장 좋은 횟수를 고른다
    history = run_passes(members, valid_rows, vocab, args, device, margin, args.n_passes)
    f1_by_pass = [mean_f1(preds, outputs) for preds in history]
    best_pass = max(range(len(history)), key=lambda i: f1_by_pass[i]) + 1
    print(
        f'앙상블({len(members)}개) 적용 횟수별 valid char F1: '
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
    print(f'valid char F1: 교정 전 {mean_f1(stage1, outputs):.4f} -> 교정 후 {f1_by_pass[best_pass - 1]:.4f} -> {valid_path}')

    # 교정이 글자 단위로 무엇을 바꿨는지 (한글 위치만)
    fixed = broken = still_wrong = 0
    for row, s, y, p in zip(valid_rows, stage1, outputs, preds):
        for xc, sc, yc, pc in zip(row['input'], s, y, p):
            if not is_hangul(xc):
                continue
            fixed += sc != yc and pc == yc
            broken += sc == yc and pc != yc
            still_wrong += sc != yc and pc != yc
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
