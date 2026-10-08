"""2단계 교정 모델: 1단계 예측문을 사전학습 KoCharELECTRA로 다시 읽어 틀린 글자를 고친다.

- 입력: 1단계 예측문의 문자 임베딩(KoCharELECTRA 사전) + 원래 난독화 글자의 초성/중성/종성 임베딩
  1단계 예측문은 대부분 정상 한국어라 사전학습 때 보던 글과 같은 형태다.
  자모 임베딩은 0으로 초기화해 학습 시작 시점에는 사전학습 모델의 입력과 완전히 같고,
  학습하면서 1단계가 버린 단서(원래 입력의 자모)를 다시 참고하게 된다.
- 인코더: monologg/kocharelectra-base-discriminator 전체 (문자 단위 토큰, 최대 512 위치)
- 출력: 한글 위치에서만 output 음절 사전으로 분류
  1단계가 예측한 음절의 logit에 copy_bias를 더해, 학습 시작 시점의 출력이 1단계 예측과 같게 한다.
  (맞는 글자를 망치지 않도록 '확신이 있을 때만 바꾸는' 쪽에서 출발)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraConfig, ElectraModel

from data_preprocessing import N_CHO, N_JUNG, N_JONG

MODEL_NAME = 'monologg/kocharelectra-base-discriminator'


def load_tokens(model_name=MODEL_NAME):
    """KoCharELECTRA의 vocab.txt를 토큰 리스트로 읽는다 (줄 번호 = 토큰 id).

    전용 토크나이저(KoCharElectraTokenizer)는 list(text)로 문자를 나눌 뿐이라 사전만 있으면 된다.
    공백도 하나의 토큰이므로 strip()이 아니라 줄바꿈만 제거해야 한다.
    """
    from huggingface_hub import hf_hub_download
    with open(hf_hub_download(model_name, 'vocab.txt'), encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


class DoubleCorrectionModel(nn.Module):
    def __init__(self, n_syllables, electra_config=None, model_name=MODEL_NAME, dropout=0.1, copy_bias=5.0):
        """electra_config(dict)가 주어지면 사전학습 가중치를 받지 않고 구조만 만든다 (체크포인트 로드용)."""
        super().__init__()
        if electra_config is None:
            self.electra = ElectraModel.from_pretrained(model_name)
        else:
            self.electra = ElectraModel(ElectraConfig.from_dict(electra_config))
        config = self.electra.config
        # 인덱스 0: 비한글/특수 토큰/패딩 (자모 없음)
        self.cho = nn.Embedding(N_CHO + 1, config.embedding_size, padding_idx=0)
        self.jung = nn.Embedding(N_JUNG + 1, config.embedding_size, padding_idx=0)
        self.jong = nn.Embedding(N_JONG + 1, config.embedding_size, padding_idx=0)
        for emb in (self.cho, self.jung, self.jong):
            nn.init.zeros_(emb.weight)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(config.hidden_size, n_syllables)
        self.copy_bias = nn.Parameter(torch.tensor(float(copy_bias)))

    def forward(self, input_ids, jamo, attention_mask, target_mask, copy_ids):
        """target_mask가 True인 위치의 logit만 (N, n_syllables)로 반환.

        input_ids: (B, L) 1단계 예측문의 토큰 id, jamo: (B, L, 3) 난독화 입력의 초성/중성/종성 인덱스,
        copy_ids: (N,) target 위치에서 1단계가 예측한 음절의 id (출력 사전에 없으면 -1).
        """
        x = (
            self.electra.embeddings.word_embeddings(input_ids)
            + self.cho(jamo[..., 0]) + self.jung(jamo[..., 1]) + self.jong(jamo[..., 2])
        )
        # 위치/타입 임베딩, LayerNorm, (크기가 다르면) hidden 차원 투영은 ELECTRA 내부에서 그대로 적용된다
        h = self.electra(inputs_embeds=x, attention_mask=attention_mask).last_hidden_state
        logits = self.head(self.dropout(h[target_mask])).float()
        has_copy = (copy_ids >= 0).float()[:, None]
        copy_onehot = F.one_hot(copy_ids.clamp(min=0), logits.size(-1)).float()
        return logits + copy_onehot * has_copy * self.copy_bias
