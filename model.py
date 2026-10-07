"""자모 임베딩 + RoPE Transformer 인코더 기반 음절 분류 모델.

- 입력: 한글 음절은 초성/중성/종성 임베딩의 합, 그 외 문자는 문자 임베딩
- 인코더: Pre-LN Transformer, 위치 정보는 RoPE(상대 위치)로 넣어 긴 문장에도 일반화
- 출력: 한글 위치에서만 output 음절 사전(약 1.5천 개)으로 분류
  (비한글 위치는 항상 그대로 복사되므로 logit을 계산하지 않아 연산을 줄임)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_preprocessing import N_CHO, N_JUNG, N_JONG, PAD

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
        d_hidden = int(2 * d_ff / 3)          # 1024 → 682, 기존 GELU FFN과 파라미터 동급
        d_hidden = (d_hidden + 63) // 64 * 64  # 704, 텐서코어 정렬
        # self.ff = nn.Sequential(
            
        #     # nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        #     nn.Linear(d_model, d_ff), SwiGLU(d_model, d_hidden, dropout), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        # )
        d_hidden = int(2 * d_ff / 3)
        d_hidden = (d_hidden + 63) // 64 * 64  # 704
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


class DeobfuscationModel(nn.Module):
    def __init__(self, n_chars, n_syllables, d_model=256, n_heads=8, n_layers=6, d_ff=1024, dropout=0.1):
        super().__init__()
        self.embedding = JamoEmbedding(n_chars, d_model, dropout)
        self.rotary = RotaryEmbedding(d_model // n_heads)
        self.layers = nn.ModuleList(EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.final_norm = nn.LayerNorm(d_model)
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
        return self.head(x[target_mask])
