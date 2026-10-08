"""앙상블용 2단계 모델을 하나 더 학습한다 (검증 분할은 그대로 두고 모델 쪽 난수만 바꿈).

double_mask_train.py / double_kc_train.py의 --seed는 검증 분할과 모델 초기화를 함께 정한다.
앙상블하려면 모든 모델이 같은 검증 문장을 써야 하므로(다르면 한 모델의 검증 문장을 다른 모델이 학습한다),
--seed는 그대로 두고 초기화, dropout, 가리는 위치에 쓰이는 난수만 --model_seed로 바꾼다.

실행:  python double_seed_train.py --model_seed 1 --seed 44 --extra_oof_path outputs_double/stage1_oof_aug.csv
       python double_seed_train.py --model_seed 1 --trainer kc --seed 44 ...   (KcELECTRA 버전)

--model_seed, --trainer 외의 인자는 고른 학습 스크립트에 그대로 넘긴다.
--ckpt_path, --output_dir을 주지 않으면 모델마다 다른 아래 이름을 쓴다.

결과물 (mask): checkpoints/double_stage2_mask_s{model_seed}.pt, outputs_double_mask_s{model_seed}/
결과물 (kc):   checkpoints/double_kc_stage2_s{model_seed}.pt,   outputs_double_kc_s{model_seed}/
"""
import argparse
import sys

from augmented_train import set_seed


def main():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--model_seed', type=int, required=True, help='모델 초기화, dropout, 가리는 위치에 쓰는 난수 seed')
    p.add_argument('--trainer', choices=['mask', 'kc'], default='mask',
                   help='mask: double_mask_train.py (KoCharELECTRA), kc: double_kc_train.py (KcELECTRA)')
    extra, rest = p.parse_known_args()
    name = 'double_stage2_mask' if extra.trainer == 'mask' else 'double_kc_stage2'
    if not any(arg.startswith('--ckpt_path') for arg in rest):
        rest += ['--ckpt_path', f'checkpoints/{name}_s{extra.model_seed}.pt']
    if not any(arg.startswith('--output_dir') for arg in rest):
        rest += ['--output_dir', f"outputs_{name.replace('_stage2', '')}_s{extra.model_seed}"]
    sys.argv = [sys.argv[0]] + rest

    if extra.trainer == 'mask':
        import double_mask_train as trainer
    else:
        import double_kc_train as trainer
    # 학습 스크립트는 시작할 때 set_seed(args.seed)를 한 번 부른다. 그 호출만 model_seed로 바꾸면
    # 검증 분할(split_rows)은 --seed를 그대로 쓰고 모델 쪽 난수만 달라진다
    trainer.set_seed = lambda _: set_seed(extra.model_seed)
    print(f'model_seed {extra.model_seed} ({extra.trainer})')
    trainer.main()


if __name__ == '__main__':
    main()
