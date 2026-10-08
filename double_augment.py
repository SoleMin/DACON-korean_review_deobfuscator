"""2단계 교정 모델의 학습 데이터를 늘린다 (1단계 재학습 없이 예측만 다시 수행).

double_stage1.py가 저장한 조각별 1단계 모델로, 그 모델이 학습 때 안 본 조각의 문장을
'새로 난독화한 입력'으로 다시 예측한다. 같은 정답 문장에서 다른 실수가 나오므로
2단계가 배울 (1단계의 실수 -> 정답) 예시가 n_variants배만큼 더 생긴다.
난독화 규칙(Obfuscator)도 조각별로 나머지 조각에서만 추정해, 예측할 조각의 실제 난독화 형태는 쓰지 않는다.

먼저:  python double_stage1.py
실행:  python double_augment.py --n_variants 3
다음:  python double_train.py --extra_oof_path outputs_double/stage1_oof_aug.csv

결과물:
- outputs_double/stage1_oof_aug.csv   추가 예측문 (ID, input, pred, output). ID는 '원래ID_obf번호'
"""
import argparse
import os

import torch

from augmented_train import char_f1, set_seed
from data_augmentation import Obfuscator
from data_preprocessing import ObfuscationDataset, Vocab, clean, read_csv, write_csv
from double_stage1 import OOF_FIELDS, ensemble_predict, load_model

AUG_SUFFIX = '_obf'  # 추가 예측문의 ID에 붙는 접미사 (뒤에 번호가 온다)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_path', default='data/train.csv')
    p.add_argument('--ckpt_dir', default='checkpoints', help='double_stage1.py의 --ckpt_dir')
    p.add_argument('--output_dir', default='outputs_double', help='double_stage1.py의 --output_dir')
    p.add_argument('--n_folds', type=int, default=5)
    p.add_argument('--n_variants', type=int, default=3, help='문장마다 새로 난독화해 예측하는 횟수')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_tokens', type=int, default=16384, help='배치당 (문장 수 x 최대 길이) 상한의 절반 (예측은 이 값의 2배를 쓴다)')
    p.add_argument('--infer_window', type=int, default=0,
                   help='예측할 때 이보다 긴 문장은 이 길이의 겹치는 구간으로 나눠 넣는다 (0이면 문장 전체를 한 번에)')
    p.add_argument('--infer_overlap', type=int, default=64, help='구간으로 나눌 때 이웃 구간이 겹치는 문자 수')
    p.add_argument('--no_amp', action='store_true', help='fp16 혼합 정밀도 끄기')
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    rows = read_csv(args.train_path)
    aug_rows = []
    for k in range(args.n_folds):
        # 조각 구성은 1단계가 남긴 파일에서 그대로 읽는다 (seed를 다르게 줘도 어긋나지 않는다)
        held_ids = {row['ID'] for row in read_csv(os.path.join(args.output_dir, f'stage1_oof_fold{k}.csv'))}
        held_rows = [row for row in rows if row['ID'] in held_ids]
        other_rows = [row for row in rows if row['ID'] not in held_ids]
        obfuscator = Obfuscator.from_rows(other_rows, args.seed + k)

        variants = []
        for row in held_rows:
            out = clean(row['output'])
            seen = {row['input']}
            for j in range(args.n_variants):
                new_inp = obfuscator(out)
                if new_inp in seen:  # 원래 입력이나 앞의 변형과 완전히 같으면 제외
                    continue
                seen.add(new_inp)
                variants.append({'ID': f"{row['ID']}{AUG_SUFFIX}{j}", 'input': new_inp, 'output': out})

        ckpt = torch.load(os.path.join(args.ckpt_dir, f'double_stage1_fold{k}.pt'), map_location=device)
        vocab = Vocab.from_state_dict(ckpt['vocab'])
        model = load_model(ckpt, device)
        dataset = ObfuscationDataset(variants, vocab)
        preds = ensemble_predict([model], dataset, vocab, args, device)
        f1s = [char_f1(p, t) for p, t in zip(preds, dataset.outputs)]
        aug_rows += [
            {'ID': i, 'input': x, 'pred': p, 'output': y}
            for i, x, p, y in zip(dataset.ids, dataset.inputs, preds, dataset.outputs)
        ]
        print(f'[조각 {k}] 새로 난독화한 문장 {len(variants)}개, char F1 {sum(f1s) / max(len(f1s), 1):.4f}')
        del model

    aug_path = os.path.join(args.output_dir, 'stage1_oof_aug.csv')
    write_csv(aug_path, aug_rows, OOF_FIELDS)
    n_wrong = sum(p != y for row in aug_rows for p, y in zip(row['pred'], row['output']))
    print(f'추가 예측문 {len(aug_rows)}개 (1단계가 틀린 글자 {n_wrong}개) -> {aug_path}')


if __name__ == '__main__':
    main()
