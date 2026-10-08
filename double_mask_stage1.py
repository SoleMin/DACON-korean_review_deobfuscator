"""1단계 모델 학습 (입력 자모 가리기 적용). double_stage1.py의 학습/예측 코드를 그대로 쓰고 학습 배치만 바꾼다.

학습 배치를 만들 때마다 한글 위치의 일부를 무작위로 골라
- char_mask_ratio만큼은 초성/중성/종성을 모두 지워, 주변 문맥만으로 정답 음절을 맞히게 하고
- jong_mask_ratio만큼은 종성만 지워, 받침이 있는지 없는지를 문맥으로 판단하게 한다.
(난독화가 받침 없는 음절의 절반가량에 가짜 받침을 붙여서, 입력의 받침은 원래 믿기 어려운 단서다.)
검증과 예측 입력은 가리지 않는다.

두 가지 방식으로 실행한다.

1) 효과 확인 (--mode check): train을 한 번만 나눠 모델 하나를 학습하고 검증 F1을 출력한다.
   가리기를 끈 실행과 켠 실행을 같은 설정으로 한 번씩 돌려 비교한다.
       python double_mask_stage1.py --mode check --valid_ratio 0.1 --char_mask_ratio 0 --jong_mask_ratio 0
       python double_mask_stage1.py --mode check --valid_ratio 0.1

2) 전체 실행 (--mode folds): double_stage1.py와 똑같이 n_folds 조각 학습 + train/test 예측문 생성까지 한다.
       python double_mask_stage1.py --mode folds

인자는 double_stage1.py의 것을 그대로 받고 --mode, --char_mask_ratio, --jong_mask_ratio만 추가된다.
--ckpt_dir, --output_dir을 주지 않으면 double_stage1.py의 결과를 덮어쓰지 않도록 아래 기본값을 쓴다.

결과물 (folds): checkpoints_mask1/double_stage1_fold{k}.pt, outputs_double_mask1/stage1_oof.csv, stage1_test.csv 등
결과물 (check): checkpoints_mask1/double_stage1_check.pt
"""
import argparse
import os
import sys

import torch

import double_stage1
from augmented_train import augment_rows, set_seed
from data_preprocessing import IGNORE_INDEX, Vocab, collate_fn, read_csv, split_rows

DEFAULT_CKPT_DIR = 'checkpoints_mask1'
DEFAULT_OUTPUT_DIR = 'outputs_double_mask1'


def parse_args():
    """이 파일 전용 인자만 먼저 떼어 내고, 나머지는 double_stage1.parse_args에 그대로 넘긴다."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--mode', choices=['check', 'folds'], default='check',
                   help='check: 한 번만 학습해 검증 F1 확인, folds: 조각별 학습 + 예측문 생성')
    p.add_argument('--char_mask_ratio', type=float, default=0.08, help='학습 때 초성/중성/종성을 모두 가리는 한글 위치의 비율')
    p.add_argument('--jong_mask_ratio', type=float, default=0.07, help='학습 때 종성만 가리는 한글 위치의 비율')
    extra, rest = p.parse_known_args()
    if not any(arg.startswith('--ckpt_dir') for arg in rest):
        rest += ['--ckpt_dir', DEFAULT_CKPT_DIR]
    if not any(arg.startswith('--output_dir') for arg in rest):
        rest += ['--output_dir', DEFAULT_OUTPUT_DIR]
    sys.argv = [sys.argv[0]] + rest
    args = double_stage1.parse_args()
    vars(args).update(vars(extra))
    return args


def make_masking_collate(args):
    def masking_collate(batch):
        """data_preprocessing.collate_fn의 결과에서 한글 위치 일부의 자모를 0(자모 정보 없음)으로 바꾼다."""
        out = collate_fn(batch)  # 매번 새 텐서라 여기서 고쳐도 데이터셋의 원본은 그대로다
        target = out['labels'] != IGNORE_INDEX
        r = torch.rand(target.shape)
        whole = target & (r < args.char_mask_ratio)
        jong_only = target & (r >= args.char_mask_ratio) & (r < args.char_mask_ratio + args.jong_mask_ratio)
        # 문자 id는 그대로 둔다 (한글이라는 사실과 패딩 여부는 유지)
        out['cho'][whole] = 0
        out['jung'][whole] = 0
        out['jong'][whole | jong_only] = 0
        return out
    return masking_collate


def make_masked_train_fold(args):
    """double_stage1.train_fold와 같되, 학습 배치를 만들 때만 가리기를 적용하는 함수를 만든다.

    train_fold는 배치를 double_stage1 모듈의 collate_fn으로 만들기 때문에, 호출하는 동안만 그 이름을 바꿔 둔다.
    학습 중 검증(augmented_train.predict)과 학습 뒤 예측(ensemble_predict)은 원래 collate_fn을 쓴다.
    """
    original_train_fold = double_stage1.train_fold
    masking_collate = make_masking_collate(args)

    def masked_train_fold(*fold_args, **fold_kwargs):
        double_stage1.collate_fn = masking_collate
        try:
            return original_train_fold(*fold_args, **fold_kwargs)
        finally:
            double_stage1.collate_fn = collate_fn
    return masked_train_fold


def check(args, device, train_fold):
    """train을 한 번만 나눠 모델 하나를 학습한다. 검증 F1은 train_fold가 epoch마다 출력한다."""
    rows = read_csv(args.train_path)
    vocab = Vocab.build(rows)
    train_rows, valid_rows = split_rows(rows, args.valid_ratio, args.seed)
    n_original = len(train_rows)
    train_rows = augment_rows(train_rows, args)
    print(f'train {len(train_rows)} (원본 {n_original} + 증강 {len(train_rows) - n_original}) / valid {len(valid_rows)}')
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train_fold(train_rows, valid_rows, vocab, args, device, os.path.join(args.ckpt_dir, 'double_stage1_check.pt'))


def main():
    args = parse_args()
    print(f'자모 가리기: 글자 전체 {args.char_mask_ratio:g} / 종성만 {args.jong_mask_ratio:g} | mode {args.mode}')
    train_fold = make_masked_train_fold(args)
    if args.mode == 'check':
        set_seed(args.seed)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'device: {device}')
        check(args, device, train_fold)
    else:
        # double_stage1.main의 조각 나누기, 이어 하기, 예측문 생성을 그대로 쓰고 학습 함수와 인자만 바꿔 끼운다
        double_stage1.train_fold = train_fold
        double_stage1.parse_args = lambda: args
        double_stage1.main()


if __name__ == '__main__':
    main()
