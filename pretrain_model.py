"""model.py의 구조 위에 사전학습 KoCharELECTRA-small의 상위 층만 얹은 음절 분류 모델.

- 입력: model.py와 같은 JamoEmbedding (한글 음절은 초성/중성/종성 임베딩의 합, 그 외 문자는 문자 임베딩)
- 하위 인코더: model.py의 RoPE Pre-LN EncoderBlock (처음부터 학습)
- 상위 인코더: monologg/kocharelectra-small-discriminator의 마지막 n_pretrained_layers개 층 (사전학습 가중치)
  ELECTRA의 임베딩과 아래쪽 층은 버린다. 위치 정보는 하위 인코더의 RoPE가 맡으므로
  ELECTRA의 절대 위치 임베딩(최대 512)을 쓰지 않아 문장 길이 제한이 없다.
- 출력: model.py와 같이 한글 위치에서만 output 음절 사전으로 분류
"""
import torch
import torch.nn as nn
from transformers import ElectraConfig, ElectraModel

from model import EncoderBlock, JamoEmbedding, RotaryEmbedding

MODEL_NAME = 'monologg/kocharelectra-small-discriminator'


class PretrainDeobfuscationModel(nn.Module):
    def __init__(self, n_chars, n_syllables, n_heads=8, n_layers=6, d_ff=1024, dropout=0.1,
                 n_pretrained_layers=4, electra_config=None, model_name=MODEL_NAME):
        """d_model은 사전학습 모델의 hidden_size(256)로 고정된다.

        electra_config(dict)가 주어지면 사전학습 가중치를 받지 않고 구조만 만든다 (체크포인트 로드용).
        이때 상위 층 수는 electra_config의 num_hidden_layers를 따른다.
        """
        super().__init__()
        if electra_config is None:
            electra = ElectraModel.from_pretrained(model_name)
            self.electra_config = electra.config
            self.pretrained = electra.encoder
            self.pretrained.layer = self.pretrained.layer[-n_pretrained_layers:]
            self.electra_config.num_hidden_layers = len(self.pretrained.layer)
        else:
            self.electra_config = ElectraConfig.from_dict(electra_config)
            self.pretrained = ElectraModel(self.electra_config).encoder
        d_model = self.electra_config.hidden_size
        self.embedding = JamoEmbedding(n_chars, d_model, dropout)
        self.rotary = RotaryEmbedding(d_model // n_heads)
        self.layers = nn.ModuleList(EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        # ELECTRA 층은 Post-LN이라 정규화된 입력을 기대한다
        self.final_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, n_syllables)

    def forward(self, char_ids, cho, jung, jong, padding_mask, target_mask):
        """target_mask가 True인 위치(한글 음절)의 logit만 (N, n_syllables)로 반환."""
        x = self.embedding(char_ids, cho, jung, jong)
        cos, sin = self.rotary(x.size(1), x.device)
        # True = attend 가능. (B, 1, 1, L)로 브로드캐스트되어 패딩 key를 가림
        attn_mask = ~padding_mask[:, None, None, :]
        for layer in self.layers:
            x = layer(x, attn_mask, cos, sin)
        x = self.final_norm(x)
        # ELECTRA 층은 attention score에 더하는 형태의 마스크를 받는다 (패딩 key = 매우 큰 음수)
        additive_mask = torch.zeros_like(attn_mask, dtype=x.dtype).masked_fill(~attn_mask, torch.finfo(x.dtype).min)
        x = self.pretrained(x, attention_mask=additive_mask).last_hidden_state
        return self.head(self.dropout(x[target_mask]))
