# -[데이콘] 난독화된 한글 리뷰 복원 및 생성 AI 경진대회-
데이콘에 등록된 경진대회 - 난독화된 한글 리뷰 복원 및 생성 AI 경진대회

난독화된 한글 리뷰를 원래 문장으로 복원한다. 입력과 정답의 글자 위치가 1:1로 대응하므로,
한글 음절 위치마다 원래 음절을 분류하는 문제로 푼다.

## 구성

```
난독화 문장 ──▶ 1단계 복원 모델 (5조각 학습) ──▶ 1차 복원문
                                                    │
난독화 문장 ──────────(자모 단서)──────────────────▶ 2단계 교정 모델 (KcELECTRA) ──▶ 반복 적용 ──▶ 최종 복원문
```

- **1단계** (`model.py`): 자모 임베딩 + RoPE Transformer 위에 KoCharELECTRA-small의 상위 층을 얹은 모델.
  train을 5조각으로 나눠 학습해, 2단계가 배울 "처음 보는 문장에 대한 예측문"을 만든다.
- **2단계** (`double_model.py`): 1단계 예측문을 KcELECTRA로 다시 읽어 틀린 글자를 고친다.
  예측문은 대부분 정상 한국어라 사전학습 지식이 그대로 쓰인다.
- **반복 적용**: 교정한 출력을 다시 교정 모델에 넣는다.

## 실행

```bash
python double_stage1.py --seed 44                                                      # 1단계 5조각 학습
python double_augment.py                                                               # 2단계 학습 데이터 확장
python double_train.py --seed 44 --extra_oof_path outputs_double/stage1_oof_aug.csv    # 2단계 학습
python double_ensemble.py --ckpt_paths checkpoints/double_stage2.pt                    # 반복 적용, 제출 파일 생성
```

데이터는 `data/train.csv`, `data/test.csv`에 둔다. 외부 데이터는 쓰지 않고, 공개된 사전학습 가중치
(`monologg/kocharelectra-small-discriminator`, `beomi/KcELECTRA-base-v2022`)만 쓴다.

구조, 학습 설정, 실험별 점수는 [result.md](result.md)에 정리했다.
