"""2단계 교정 모델 (KcELECTRA 버전): 1단계 예측문을 댓글로 사전학습된 KcELECTRA로 읽고 글자 단위로 고친다.

KcELECTRA는 글자 단위가 아니라 서브워드(WordPiece) 단위 모델이라, double_model.py처럼 글자를 그대로 넣을 수 없다.
그래서 인코더는 1단계 예측문을 원래 방식대로 서브워드로 읽게 하고, 그 위에 글자 단위 층을 얹는다.

- 인코더: beomi/KcELECTRA-base-v2022 전체. 입력은 1단계 예측문의 서브워드 토큰 (사전학습 때와 같은 형태)
- 글자 표현: 각 글자가 속한 서브워드 토큰의 출력
              + 원래 난독화 글자의 자모 임베딩 + 1단계 예측 글자의 자모 임베딩 (model.py의 JamoEmbedding)
- 글자 단위 층: model.py의 RoPE Transformer 블록 (처음부터 학습). 서브워드 문맥과 글자별 단서를 섞는다
- 출력: 한글 위치에서만 output 음절 사전으로 분류
  double_model.py와 같이 1단계가 예측한 음절의 logit에 copy_bias를 더해, 1단계 예측에서 출발한다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraConfig, ElectraModel

from data_preprocessing import PAD
from model import EncoderBlock, JamoEmbedding, RotaryEmbedding

MODEL_NAME = 'beomi/KcELECTRA-base-v2022'


class DoubleKcModel(nn.Module):
    def __init__(self, n_chars, n_syllables, electra_config=None, model_name=MODEL_NAME,
                 d_model=384, n_heads=8, n_layers=2, d_ff=1536, dropout=0.1, copy_bias=5.0):
        """electra_config(dict)가 주어지면 사전학습 가중치를 받지 않고 구조만 만든다 (체크포인트 로드용)."""
        super().__init__()
        if electra_config is None:
            self.electra = ElectraModel.from_pretrained(model_name)
        else:
            self.electra = ElectraModel(ElectraConfig.from_dict(electra_config))
        self.proj = nn.Linear(self.electra.config.hidden_size, d_model)
        self.obf_embedding = JamoEmbedding(n_chars, d_model, dropout)   # 원래 난독화 글자
        self.pred_embedding = JamoEmbedding(n_chars, d_model, dropout)  # 1단계 예측 글자
        self.rotary = RotaryEmbedding(d_model // n_heads)
        self.layers = nn.ModuleList(EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_syllables)
        self.copy_bias = nn.Parameter(torch.tensor(float(copy_bias)))

    def forward(self, input_ids, attention_mask, char2tok, obf, pred, target_mask, copy_ids):
        """target_mask가 True인 위치의 logit만 (N, n_syllables)로 반환.

        input_ids, attention_mask: (B, T) 1단계 예측문의 서브워드 토큰
        char2tok: (B, L) 각 글자가 속한 토큰의 위치. 어느 토큰에도 속하지 않으면(공백, 패딩) 0
        obf, pred: (B, L, 4) 난독화 글자 / 1단계 예측 글자의 (문자 id, 초성, 중성, 종성)
        copy_ids: (N,) target 위치에서 1단계가 예측한 음절의 id (출력 사전에 없으면 -1)
        """
        h = self.electra(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # 글자마다 자기가 속한 토큰의 출력을 가져온다. 토큰이 없는 글자는 0 벡터
        h = h.gather(1, char2tok[..., None].expand(-1, -1, h.size(-1)))
        h = self.proj(h) * (char2tok > 0)[..., None]
        x = h + self.obf_embedding(*obf.unbind(-1)) + self.pred_embedding(*pred.unbind(-1))
        cos, sin = self.rotary(x.size(1), x.device)
        # True = attend 가능. (B, 1, 1, L)로 브로드캐스트되어 패딩 key를 가림
        attn_mask = (obf[..., 0] != PAD)[:, None, None, :]
        for layer in self.layers:
            x = layer(x, attn_mask, cos, sin)
        logits = self.head(self.final_norm(x)[target_mask]).float()
        has_copy = (copy_ids >= 0).float()[:, None]
        copy_onehot = F.one_hot(copy_ids.clamp(min=0), logits.size(-1)).float()
        return logits + copy_onehot * has_copy * self.copy_bias
