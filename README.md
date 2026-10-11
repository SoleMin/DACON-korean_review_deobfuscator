# -[데이콘] 난독화된 한글 리뷰 복원 및 생성 AI 경진대회-

![Vibe Coded](https://img.shields.io/badge/Vibe-Coded-blueviolet)

데이콘에 등록된 경진대회 - 난독화된 한글 리뷰 복원 및 생성 AI 경진대회


대회 링크 : https://dacon.io/competitions/official/236446/overview/description

## 1. 요약

| 항목 | 내용 |
|---|---|
| 과제 | 난독화된 한글 숙박 리뷰를 원래 문장으로 복원 (Dacon) |
| 평가 | 문자 단위 F1 Score|
| 접근(aproach) | 위치별 음절 분류 모델(1단계)의 출력을, 사전학습 언어 모델(2단계)이 다시 읽고 고치는 2단계 구조, 추가 보정을 위해 2단계 모델의 출력물을 2단계 모델의 입력으로 사용하는 과정 n회를 추가 |
| 리더보드 최고점 | **0.989**(1단계 5 fold + KcELECTRA 2단계 + 3회 반복 적용)|
| 베이스라인 | 0.977 (augmentation을 적용한 단일 모델(사전학습모델 x)) |

외부 데이터는 쓰지 않았고, 공개된 사전학습 가중치 두 개만 사용
(`monologg/kocharelectra-small-discriminator`, `beomi/KcELECTRA-base-v2022`).
test 데이터는 예측에만 쓴다.

## 2. 전체 구조

```
train.csv ──▶ [1단계] 5 fold 학습 ──▶ stage1_oof.csv      train 전체의 '처음 보는 문장' 예측문
                  │               └▶ stage1_test.csv      test 예측문 (5개 모델 확률 평균)
                  │
                  └▶ [추가 예측] 새로 난독화한 문장을 fold별 모델로 예측 ──▶ stage1_oof_aug.csv
                                                                              │
stage1_oof.csv + stage1_oof_aug.csv ──▶ [2단계] 교정 모델 학습 ◀───────────────┘
                                              │
stage1_test.csv ──────────────────────────────┴──▶ [반복 적용] ──▶ submission.csv
```

입력과 정답의 글자가 1:1로 대응하므로, 한글 음절 위치마다 원래 음절을 분류하는 문제로 푼다.

1. **1단계**: 난독화 문장을 복원한다. train을 5 fold로 나눠, 각 모델이 학습 때 안 본 문장을 예측한 결과를 모은다.
2. **추가 예측**: 정답 문장을 새로 난독화해 1단계 모델로 예측한다. 2단계 학습 데이터를 늘리기 위한 것이다.
3. **2단계**: 1단계의 복원문을 사전학습 모델(KcELECTRA)로 다시 읽어 틀린 글자를 고친다.
4. **반복 적용**: 2단계의 출력을 다시 2단계에 넣어 한 번 더 고친다.

## 3. 구성 요소

### 1단계: 복원 모델 (`stage1_model.py`)

| 순서 | 구성 | 설명 |
|---|---|---|
| 1 | 자모 임베딩 | 한글은 초성 + 중성 + 종성 임베딩의 합, 그 외 글자는 문자 임베딩 |
| 2 | RoPE Transformer 6층 | 처음부터 학습 |
| 3 | KoCharELECTRA-small 상위 6층 | 사전학습 가중치 |
| 4 | 분류 헤드 | 한글 위치에서만 출력 음절을 분류 |

### Augmentation (`augmentation.py`, 1단계 학습 데이터에 적용)

| 종류 | 내용 |
|---|---|
| 부분 복원 | 입력의 일부 단어를 정답 단어로 되돌림 |
| 단어 사전 치환 | 같은 정답 단어의 다른 난독화 형태로 교체 |
| 재난독화 | 데이터에서 추정한 자모 규칙으로 정답을 새로 난독화 |
| 이어 붙이기 | 문장들을 이어 긴 샘플 생성 |

### 2단계: 교정 모델 (`stage2_model.py`)

| 순서 | 구성 | 설명 |
|---|---|---|
| 1 | KcELECTRA-base | 1단계 복원문을 읽는다 |
| 2 | 글자 표현 | 각 글자가 속한 토큰의 출력 + 난독화 글자의 자모 임베딩 + 1단계 예측 글자의 자모 임베딩 |
| 3 | RoPE Transformer 2층 | 문맥과 글자별 단서를 섞는다 |
| 4 | 분류 헤드 | 한글 위치마다 최종 음절을 분류 |

학습할 때는 1단계 예측 글자의 일부를 가리거나 오답으로 바꿔, 고치는 예시를 늘린다 (`stage2_data.py`).

## 4. 결과

| 구성 | 검증 F1 | 리더보드 |
|---|---|---|
| 1단계만 | 0.9832 | |
| + KcELECTRA 2단계 | 0.9886 | 0.9863 |
| + 반복 적용 | 0.9893 | 0.9870 |
| + KoCharELECTRA 2단계 모델과 앙상블 | 0.9901 | 0.98846 |
| + 여러 구성의 제출 파일을 글자별 다수결 | | **0.989** |

이 저장소에는 위에서 세 번째 행까지의 코드가 있다.

## 5. 실행 방법

```bash
python stage1_train.py --seed 44                                                # 1단계 5 fold 학습
python stage1_predict_aug.py                                                    # 추가 예측문 생성
python stage2_train.py --seed 44 --extra_oof_path outputs/stage1_oof_aug.csv    # 2단계 학습
python predict.py --ckpt_paths outputs/stage2.pt                                # 반복 적용, 제출 파일 생성
```

- 데이터는 `data/train.csv`, `data/test.csv`에 둔다.
- 모델과 예측 결과는 `outputs/` 아래에 만들어지고, 최종 제출 파일은 `outputs/submission.csv`다.

| 파일 | 역할 |
|---|---|
| `preprocessing.py` | CSV 입출력, 자모 분해, 사전, 데이터셋 |
| `augmentation.py` | Augmentation 4종 |
| `trainer.py` | 공용 학습 루프 |
| `stage1_model.py`, `stage1_train.py`, `stage1_predict_aug.py` | 1단계 모델, 5 fold 학습, 추가 예측문 |
| `stage2_model.py`, `stage2_data.py`, `stage2_train.py` | 2단계 모델, 데이터셋, 학습 |
| `predict.py` | 반복 적용, 제출 파일 생성 |
