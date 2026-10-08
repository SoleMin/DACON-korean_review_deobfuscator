"""1단계 복원 모델: 자모 임베딩 + RoPE Transformer 위에 사전학습 KoCharELECTRA-small의 상위 층을 얹은 음절 분류 모델.

- 입력: 한글 음절은 초성/중성/종성 임베딩의 합, 그 외 문자는 문자 임베딩
- 하위 인코더: Pre-LN Transformer (처음부터 학습). 위치 정보는 RoPE(상대 위치)로 넣어 문장 길이 제한이 없다
- 상위 인코더: monologg/kocharelectra-small-discriminator의 마지막 n_pretrained_layers개 층 (사전학습 가중치)
  ELECTRA의 임베딩과 아래쪽 층은 버린다
- 출력: 한글 위치에서만 output 음절 사전(약 1.4천 개)으로 분류
  (비한글 위치는 항상 그대로 복사되므로 logit을 계산하지 않아 연산을 줄임)

JamoEmbedding, RotaryEmbedding, EncoderBlock은 2단계 교정 모델(double_model.py)도 함께 쓴다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ElectraConfig, ElectraModel

from data_preprocessing import N_CHO, N_JUNG, N_JONG, PAD

MODEL_NAME = 'monologg/kocharelectra-small-discriminator'


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_hidden, dropout):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_hidden, bias=False)  # gate
        self.w2 = nn.Linear(d_model, d_hidden, bias=False)  # value
        self.w3 = nn.Linear(d_hidden, d_model, bias=False)  # down
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w3(self.dropout(F.silu(self.w1(x)) * self.w2(x)))


class JamoEmbedding(nn.Module):
    def __init__(self, n_chars, d_model, dropout):
        super().__init__()
        # 인덱스 0: 비한글/패딩 (자모 없음)
        self.char = nn.Embedding(n_chars, d_model, padding_idx=PAD)
        self.cho = nn.Embedding(N_CHO + 1, d_model, padding_idx=0)
        self.jung = nn.Embedding(N_JUNG + 1, d_model, padding_idx=0)
        self.jong = nn.Embedding(N_JONG + 1, d_model, padding_idx=0)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids, cho, jung, jong):
        x = self.char(char_ids) + self.cho(cho) + self.jung(jung) + self.jong(jong)
        return self.dropout(self.norm(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        return freqs.cos()[None, None], freqs.sin()[None, None]  # (1, 1, L, head_dim/2)


def apply_rotary(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2).type_as(x)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attn_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff_norm = nn.LayerNorm(d_model)
        # SwiGLU는 가중치가 3개라 중간 차원을 2/3로 줄여 GELU FFN과 파라미터 수를 맞춘다 (1024 -> 682)
        d_hidden = int(2 * d_ff / 3)
        d_hidden = (d_hidden + 63) // 64 * 64  # 64의 배수로 올림 (704)
        self.ff = SwiGLU(d_model, d_hidden, dropout)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x, attn_mask, cos, sin):
        bsz, seq_len, d_model = x.shape
        q, k, v = (
            self.qkv(self.attn_norm(x))
            .view(bsz, seq_len, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
        # PyTorch SDPA: 가능한 경우 메모리 효율적인 fused attention 커널 사용
        h = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.attn_dropout if self.training else 0.0
        )
        h = h.transpose(1, 2).reshape(bsz, seq_len, d_model)
        x = x + self.dropout(self.out_proj(h))
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x


class Stage1Model(nn.Module):
    def __init__(self, n_chars, n_syllables, n_heads=8, n_layers=6, d_ff=1024, dropout=0.1,
                 n_pretrained_layers=6, electra_config=None, model_name=MODEL_NAME):
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
