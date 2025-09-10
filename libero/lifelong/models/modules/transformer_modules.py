import math
import numpy as np
from torch import nn
import torch
import torchvision
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


###############################################################################
#
# Building blocks for transformers
#
###############################################################################


class Norm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x)


class Attention_Skill(nn.Module):  # MultiHead (Cross) Attention
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0):
        super().__init__()

        self.att_weights = None
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)

        self.q = nn.Linear(dim, num_heads * head_output_size, bias=False)
        self.k = nn.Linear(dim, num_heads * head_output_size, bias=False)
        self.v = nn.Linear(dim, num_heads * head_output_size, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )

    def forward(self, input_q, input_k, input_v, mask=None):
        B_q, N_q, C_q = input_q.shape
        B_k, N_k, C_k = input_k.shape
        B_v, N_v, C_v = input_v.shape

        q = self.q(input_q).reshape(B_q, N_q, self.num_heads, -1).transpose(1, 2)
        k = self.k(input_k).reshape(B_k, N_k, self.num_heads, -1).transpose(1, 2)
        v = self.v(input_v).reshape(B_v, N_v, self.num_heads, -1).transpose(1, 2)
        # print("q: ", q.shape)  (B, num_heads:6, 50, 64)
        # print("k: ", k.shape)  (B, num_heads:6, 100, 64)
        # print("v: ", v.shape)

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale  # (B, num_heads:6, 50, 100)
        if mask is not None:  # (B, 50, 100)
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                print("attention 维度：", attn.shape)
                print("mask 维度：", mask.shape)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                    len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn
        # print("att: ", attn.shape)

        # (..., num_heads, seq_len, head_output_size)
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")  # (B:32, T*num_modalities:50, 384)
        # print("out shape: ", out.shape)
        return self.output_layer(out)


class CP_Attention_Skill(nn.Module):  # MultiHead (Cross) Attention
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0, R=64):
        super().__init__()

        self.att_weights = None
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.s = 1
        self.R = R
        self.CP_attention = nn.Parameter(torch.zeros(self.R, 4))  # P
        nn.init.xavier_uniform_(self.CP_attention)
        self.dp = nn.Dropout(dropout)

        self.q = nn.Linear(dim, num_heads * head_output_size, bias=False)
        self.k = nn.Linear(dim, num_heads * head_output_size, bias=False)
        self.v = nn.Linear(dim, num_heads * head_output_size, bias=False)
        self.proj = nn.Linear(dim, dim)
        # self.proj = nn.Linear(dim * num_heads, dim)  # 改了输入维度
        # self.proj1 = nn.Linear(dim * num_heads * num_heads, dim)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )

    def forward(self, input_q, input_k, input_v, mask=None, CP_U=None, CP_V=None, CP_C=None):
        B_q, N_q, C_q = input_q.shape
        B_k, N_k, C_k = input_k.shape
        B_v, N_v, C_v = input_v.shape

        CPc = CP_C @ self.CP_attention  # (R * R * 4)
        q_cp, k_cp, v_cp, proj_cp = CPc[:, :, 0], CPc[:, :, 1], CPc[:, :, 2], CPc[:, :, 3]

        q = self.q(input_q)
        q += CP_V(self.dp(CP_U(input_q) @ q_cp))
        # q (B, num_heads:8, T*num_modalities:50, num_heads * head_output_size:512)
        q = q.reshape(B_q, N_q, self.num_heads, -1).transpose(1, 2)
        k = self.k(input_k)
        k += CP_V(self.dp(CP_U(input_k) @ k_cp))
        k = k.reshape(B_k, N_k, self.num_heads, -1).transpose(1, 2)
        v = self.v(input_v)
        v += CP_V(self.dp(CP_U(input_v) @ v_cp))
        # v (B, num_heads:8, num_codes:100, head_output_size:64)
        v = v.reshape(B_v, N_v, self.num_heads, -1).transpose(1, 2)
        # print("q: ", q.shape)
        # print("k: ", k.shape)
        # print("v: ", v.shape)

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale  # (B:32, num_heads:6, T*num_modalities:50, num_codes:100)
        # print("att: ", attn.shape)
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N_q:50, N_k:100)?
                # print("mask shape: ", mask.shape)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                    len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        x = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")  # (B, T*num_modalities:50, E:384)
        proj = self.proj(x)
        proj += CP_V(self.dp(CP_U(x) @ proj_cp)) * self.s

        # x = torch.matmul(attn, v)  # [B, 50, embed_size, num_heads]
        # proj = self.proj(x)
        # proj += CP_V(self.dp(CP_U(x) @ proj_cp)) * self.s
        # proj = rearrange(proj, "b h n d -> b n (h d)")  # 增加了维度变化操作
        # proj = self.proj1(proj)  # 多加了一层liner

        # x = torch.matmul(attn, v)  # [B, 50, num_heads, embed_size]
        # proj = CP_V(self.dp(CP_U(x) @ proj_cp)) * self.s
        # proj = rearrange(proj, "b h n d -> b n (h d)")  # 增加了维度变化操作
        # proj = self.proj(proj)
        # x = rearrange(x, "b h n d -> b n (h d)")  # 增加了维度变化操作
        # proj += self.proj(x)
        return self.dp(proj)


class Attention_Prompt(nn.Module):  # MultiHead (Self) Attention
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0):
        super().__init__()

        self.att_weights = None
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim, num_heads * head_output_size * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, prompt=None):
        B, N, C = x.shape  # (B, 50, 64)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        # print("k", k.shape)  # (B, num_heads:6, 50, 64)

        if prompt is not None:
            pk, pv = prompt  # (B, 50, 1, 64)

            pk = pk.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            pv = pv.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
            # print('pk', pk.shape)
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
            # print("k", k.shape)  # (B, num_heads:8, 100, 64)
            # print("v", v.shape)  # (B, num_heads:8, 100, 64)

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale  # (B, num_heads:8, 50, 100)
        # print("prompt_attention 维度：", attn.shape)
        # print("prompt_mask 维度：", mask.shape)
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                    len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.output_layer(out)


class CP_Attention_Skill_Prompt(nn.Module):  # MultiHead (Cross) Attention
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0, R=64):
        super().__init__()

        self.att_weights = None
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.s = 1
        self.R = R
        self.CP_attention = nn.Parameter(torch.zeros(self.R, 4))  # P
        nn.init.xavier_uniform_(self.CP_attention)
        self.dp = nn.Dropout(dropout)

        self.q = nn.Linear(dim, num_heads * head_output_size, bias=False)
        self.k = nn.Linear(dim, num_heads * head_output_size, bias=False)
        self.v = nn.Linear(dim, num_heads * head_output_size, bias=False)
        self.proj = nn.Linear(dim, dim)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )

    def forward(self, input_q, input_k, input_v, mask=None, CP_U=None, CP_V=None, CP_C=None, prompt=None):
        B_q, N_q, C_q = input_q.shape
        B_k, N_k, C_k = input_k.shape
        B_v, N_v, C_v = input_v.shape

        CPc = CP_C @ self.CP_attention  # (R * R * 4)
        q_cp, k_cp, v_cp, proj_cp = CPc[:, :, 0], CPc[:, :, 1], CPc[:, :, 2], CPc[:, :, 3]

        q = self.q(input_q)
        q += CP_V(self.dp(CP_U(input_q) @ q_cp))
        # q (B, num_heads:8, T*num_modalities:50, num_heads * head_output_size:512)
        q = q.reshape(B_q, N_q, self.num_heads, -1).transpose(1, 2)
        k = self.k(input_k)
        k += CP_V(self.dp(CP_U(input_k) @ k_cp))
        k = k.reshape(B_k, N_k, self.num_heads, -1).transpose(1, 2)
        v = self.v(input_v)
        v += CP_V(self.dp(CP_U(input_v) @ v_cp))
        # v (B, num_heads:8, num_codes:100, head_output_size:64)
        v = v.reshape(B_v, N_v, self.num_heads, -1).transpose(1, 2)

        if prompt is not None:
            pk, pv = prompt  # (B, T*num_modalities:50, e_p_length/2, embed_size)
            #  TODO(XJK): 原论文8是序列长度，但是这里被用作num_heads拼接起来了，序列长度为50

            pk = pk.reshape(B_k, -1, self.num_heads, C_k // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B_v, -1, self.num_heads, C_v // self.num_heads).permute(0, 2, 1, 3)
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
            # print("k", k.shape)  # (B, num_heads:8, 100, num_heads * head_output_size:512)
            # print("v", v.shape)  # (B, num_heads:8, 100, num_heads * head_output_size:512)

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale  # (B:32, num_heads:8, T*num_modalities:50, num_codes:100)
        # print("att: ", attn.shape)
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N_q:50, N_k:100)?
                # print("mask shape: ", mask.shape)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                    len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        x = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        proj = self.proj(x)
        proj += CP_V(self.dp(CP_U(x) @ proj_cp)) * self.s
        return self.dp(proj)


class Attention_Action(nn.Module):  # MultiHead (Self) Attention
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0):
        super().__init__()

        self.att_weights = None
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim, num_heads * head_output_size * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                    len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.output_layer(out)


class CP_Attention_Action(nn.Module):  # MultiHead (Self) Attention
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0, R=64):
        super().__init__()

        self.att_weights = None
        self.num_heads = num_heads
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim, num_heads * head_output_size * 3, bias=False)
        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )
        self.proj = nn.Linear(dim, dim)

        self.R = R
        self.CP_attention = nn.Parameter(torch.zeros(self.R, 4))  # P
        nn.init.xavier_uniform_(self.CP_attention)
        self.dp = nn.Dropout(dropout)
        self.s = 1

    def forward(self, x, mask=None, CP_U=None, CP_V=None, CP_C=None):
        B, N, C = x.shape
        qkv = self.qkv(x)

        CPc = CP_C @ self.CP_attention  # (R * R * 4)
        q_cp, k_cp, v_cp, proj_cp = CPc[:, :, 0], CPc[:, :, 1], CPc[:, :, 2], CPc[:, :, 3]

        q = CP_V(self.dp(CP_U(x) @ q_cp))
        k = CP_V(self.dp(CP_U(x) @ k_cp))
        v = CP_V(self.dp(CP_U(x) @ v_cp))
        qkv += torch.cat([q, k, v], dim=2)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
        # print("attention 维度：", attn.shape)
        # print("mask 维度：", mask.shape)
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                    len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        x = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        proj = self.proj(x)
        proj += CP_V(self.dp(CP_U(x) @ proj_cp)) * self.s
        return self.dp(proj)


class Attention(nn.Module):  # MultiHead (Self) Attention
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0):
        super().__init__()

        self.att_weights = None
        self.num_heads = num_heads
        # \sqrt{d_{k}}
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim, num_heads * head_output_size * 3, bias=False)

        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(num_heads * head_output_size, dim), nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
        # print("attention 维度：", attn.shape)  # (1, 6, 50, 50)
        # print("mask 维度：", mask.shape)  # (1, 50, 50)
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                    len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        out = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")
        return self.output_layer(out)


class CP_Attention(nn.Module):  # MultiHead (Self) Attention
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0, R=64):
        super().__init__()

        self.att_weights = None
        self.num_heads = num_heads
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim, num_heads * head_output_size * 3, bias=False)
        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(head_output_size * num_heads, dim), nn.Dropout(dropout)
        )
        self.proj = nn.Linear(dim, dim)
        # self.proj = nn.Linear(dim * num_heads, dim)  # 改了输入维度  * num_heads
        # self.proj1 = nn.Linear(dim * num_heads * num_heads, dim)

        self.R = R
        self.CP_attention = nn.Parameter(torch.zeros(self.R, 4))  # P
        nn.init.xavier_uniform_(self.CP_attention)
        self.dp = nn.Dropout(dropout)
        self.s = 1

    def forward(self, x, mask=None, CP_U=None, CP_V=None, CP_C=None):
        B, N, C = x.shape
        qkv = self.qkv(x)

        CPc = CP_C @ self.CP_attention  # (R * R * 4)
        q_cp, k_cp, v_cp, proj_cp = CPc[:, :, 0], CPc[:, :, 1], CPc[:, :, 2], CPc[:, :, 3]

        q = CP_V(self.dp(CP_U(x) @ q_cp))  # (B, 50, embed_size)
        k = CP_V(self.dp(CP_U(x) @ k_cp))  # (B, 50, embed_size)
        v = CP_V(self.dp(CP_U(x) @ v_cp))  # (B, 50, embed_size)
        # print("q shape: ", q.shape)
        # print("k shape: ", k.shape)
        # print("v shape: ", v.shape)
        # print("qkv shape: ", qkv.shape)
        qkv += torch.cat([q, k, v], dim=2)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
        # print("attention 维度：", attn.shape)
        # print("mask 维度：", mask.shape)
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                # print("CP Attention")
                # print("attention 维度：", attn.shape)
                # print("mask 维度：", mask.shape)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                    len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        x = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")  # [B, 50, embed_size*num_heads]
        # x = torch.matmul(attn, v)  # [B, 50, num_heads, embed_size]
        proj = CP_V(self.dp(CP_U(x) @ proj_cp)) * self.s
        # proj += CP_V(self.dp(CP_U(x) @ proj_cp)) * self.s
        # proj = rearrange(proj, "b h n d -> b n (h d)")  # 增加了维度变化操作
        # proj = self.proj(proj)
        # x = rearrange(x, "b h n d -> b n (h d)")  # 增加了维度变化操作
        proj += self.proj(x)
        # proj = self.proj1(proj)  # 多加了一层liner
        return self.dp(proj)


class Hier_CP_Attention(nn.Module):  # MultiHead (Self) Attention
    def __init__(self, dim, num_heads=8, head_output_size=64, dropout=0.0, R=64):
        super().__init__()

        self.att_weights = None
        self.num_heads = num_heads
        self.att_scale = head_output_size ** (-0.5)
        self.qkv = nn.Linear(dim, num_heads * head_output_size * 3, bias=False)
        # We need to combine the output from all heads
        self.output_layer = nn.Sequential(
            nn.Linear(head_output_size * num_heads, dim), nn.Dropout(dropout)
        )
        # self.proj = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)  # 改了输入维度  * num_heads

        self.R = R
        self.CP_attention = nn.Parameter(torch.zeros(self.R, 4))  # P
        nn.init.xavier_uniform_(self.CP_attention)
        self.dp = nn.Dropout(dropout)
        self.s = 1

    def forward(self, x, mask=None, CP_U=None, CP_V=None, CP_C=None):
        B, N, C = x.shape
        qkv = self.qkv(x)

        CPc = CP_C @ self.CP_attention  # (R * R * 4)
        q_cp, k_cp, v_cp, proj_cp = CPc[:, :, 0], CPc[:, :, 1], CPc[:, :, 2], CPc[:, :, 3]

        q = CP_V(self.dp(CP_U(x) @ q_cp))  # (B, 50, embed_size)
        k = CP_V(self.dp(CP_U(x) @ k_cp))  # (B, 50, embed_size)
        v = CP_V(self.dp(CP_U(x) @ v_cp))  # (B, 50, embed_size)
        # print("q shape: ", q.shape)
        # print("k shape: ", k.shape)
        # print("v shape: ", v.shape)
        # print("qkv shape: ", qkv.shape)
        qkv += torch.cat([q, k, v], dim=2)

        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # q.dot(k.transpose)
        attn = (q @ k.transpose(-2, -1)) * self.att_scale
        # print("attention 维度：", attn.shape)
        # print("mask 维度：", mask.shape)
        if mask is not None:
            mask = mask.bool()
            if len(mask.shape) == 2:  # (B, N)
                attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
            elif len(mask.shape) == 3 and mask.shape[0] == 1:  # (1, N, N)
                # print("CP Attention")
                # print("attention 维度：", attn.shape)
                # print("mask 维度：", mask.shape)
                attn = attn.masked_fill(~mask[None, :, :, :], float("-inf"))
            elif (
                    len(mask.shape) == 3
            ):  # Consider the case where each batch has different causal mask, typically useful for MAE implementation
                attn = attn.masked_fill(
                    ~mask[:, None, :, :].repeat(1, self.num_heads, 1, 1), float("-inf")
                )
            else:
                raise Exception("mask shape is not correct for attention")
        attn = attn.softmax(dim=-1)
        self.att_weights = attn

        # (..., num_heads, seq_len, head_output_size)
        x = rearrange(torch.matmul(attn, v), "b h n d -> b n (h d)")  # [B, 50, embed_size*num_heads]
        proj = self.proj(x)
        proj += CP_V(self.dp(CP_U(x) @ proj_cp)) * self.s
        return self.dp(proj)


class TransformerFeedForwardNN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        # Remember the residual connection
        layers = [
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CPTransformerFeedForwardNN(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0, R=64, num_heads=6):
        super().__init__()
        # Remember the residual connection
        self.fc1 = nn.Linear(dim, hidden_dim)  # 输出改为了输入维度的6倍
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        # self.fc3 = nn.Linear(hidden_dim * 6, dim)
        self.dp = nn.Dropout()
        self.drop = nn.Dropout(dropout)
        self.R = R
        self.s = 1
        self.mlp_CP = nn.Parameter(torch.zeros(self.R, 8))
        # self.mlp_CP = nn.Parameter(torch.zeros(self.R, 2))
        nn.init.xavier_uniform_(self.mlp_CP)

    def forward(self, x, CP_U=None, CP_V=None, CP_C=None):
        B, N, C = x.shape
        CPc = CP_C @ self.mlp_CP
        fc1_cp, fc2_cp = CPc[:, :, :4].reshape(self.R, self.R * 4), CPc[:, :, 4:].reshape(self.R, self.R * 4)
        # fc1_cp, fc2_cp = CPc[:, :, :1].reshape(self.R, self.R), CPc[:, :, 1:].reshape(self.R, self.R)
        h = self.fc1(x)  # (B, N, 6E) 6的意思是MLP隐层的维度是embed size的六倍
        h += CP_V(self.dp(CP_U(x) @ fc1_cp).reshape(
            B, N, 4, self.R)).reshape(
            B, N, 4 * C) * self.s
        # h += CP_V(self.dp(CP_U(x) @ fc1_cp)) * self.s
        x = self.act(h)
        x = self.drop(x)
        h = self.fc2(x)  # (B,N,E)
        x = x.reshape(B, N, 4, C)
        # x = x.reshape(B, N, 6, C)
        h += CP_V(self.dp(CP_U(x).reshape(
            B, N, 4 * self.R) @ fc2_cp.t())) * self.s
        # h += self.fc3(CP_V(self.dp(CP_U(x) @ fc2_cp)).reshape(
        #     B, N, 6 * self.R * 6)) * self.s
        x = self.drop(h)
        return x


def drop_path(
        x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, input_size, inv_freq_factor=10, factor_ratio=None):
        super().__init__()
        self.input_size = input_size
        self.inv_freq_factor = inv_freq_factor
        channels = self.input_size
        channels = int(np.ceil(channels / 2) * 2)

        inv_freq = 1.0 / (
                self.inv_freq_factor ** (torch.arange(0, channels, 2).float() / channels)
        )
        self.channels = channels
        self.register_buffer("inv_freq", inv_freq)

        if factor_ratio is None:
            self.factor = 1.0
        else:
            factor = nn.Parameter(torch.ones(1) * factor_ratio)
            self.register_parameter("factor", factor)

    def forward(self, x):
        pos_x = torch.arange(x.shape[1], device=x.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)  # 矩阵（向量）乘法，字符串表示维度变化
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        return emb_x * self.factor

    def output_shape(self, input_shape):
        return input_shape

    def output_size(self, input_size):
        return input_size


###############################################################################
#
# Transformer Decoder (we only use transformer decoder for our policies)
#
###############################################################################


class TransformerDecoder_Skill(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            num_heads,
            head_output_size,
            mlp_hidden_size,
            dropout,
            **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.attention_output = {}
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        Attention(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                        ),
                        Attention_Skill(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                        ),
                        Norm(input_size),
                        TransformerFeedForwardNN(
                            input_size, mlp_hidden_size, dropout=dropout,
                        ),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

        self.codebook_num_elements = None
        self.codebook_mask = None

    def compute_mask(self, input_shape, codebook_shape):
        # input_shape = (:, seq_len, num_elements)
        if (
                (self.num_elements is None)
                or (self.seq_len is None)
                or (self.num_elements != input_shape[2])
                or (self.seq_len != input_shape[1])
        ):
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

        if (
                (self.codebook_num_elements is None)
                or (self.codebook_num_elements != codebook_shape[1])
        ):
            self.codebook_num_elements = codebook_shape[1]
            self.codebook_original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.codebook_mask = 1 - self.codebook_original_mask.repeat_interleave(
                self.codebook_num_elements // self.seq_len, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)

    def forward(self, x, memory, mask=None, codebook_mask=None):
        for layer_idx, (att_norm, self_att, cross_att, ff_norm, ff) in enumerate(self.layers):
            if mask is not None and codebook_mask is not None:
                x = x + drop_path(self_att(att_norm(x), mask))  # Self Attention
                mha_out = cross_att(att_norm(x), att_norm(memory), att_norm(memory), mask)
                self.attention_output[layer_idx] = cross_att.att_weights
                x = x + drop_path(mha_out)
            elif self.mask is not None and self.codebook_mask is not None:
                # print("self.mask的维度：", self.mask.shape)
                # print("self.codebook_mask的维度：", self.codebook_mask.shape)
                x = x + drop_path(self_att(att_norm(x), self.mask))
                mha_out = cross_att(att_norm(x), att_norm(memory), att_norm(memory), self.codebook_mask)
                self.attention_output[layer_idx] = cross_att.att_weights
                x = x + drop_path(mha_out)
            else:  # no masking, just use full attention
                x = x + drop_path(self_att(att_norm(x)))
                mha_out = cross_att(att_norm(x), att_norm(memory), att_norm(memory))
                self.attention_output[layer_idx] = cross_att.att_weights
                x = x + drop_path(mha_out)

            # if not self.training:
            #     self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class CP_TransformerDecoder_Skill(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            num_heads,
            head_output_size,
            mlp_hidden_size,
            dropout,
            R,
            **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.attention_output = {}
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        CP_Attention(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                            R=R,
                        ),
                        CP_Attention_Skill(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                            R=R,
                        ),
                        Norm(input_size),
                        CPTransformerFeedForwardNN(
                            input_size, mlp_hidden_size, dropout=dropout, R=R, num_heads=num_heads),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

    def compute_mask(self, input_shape):
        # input_shape = (:, seq_len, num_elements)
        if (
                (self.num_elements is None)
                or (self.seq_len is None)
                or (self.num_elements != input_shape[2])  # 5
                or (self.seq_len != input_shape[1])  # 10
        ):
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements
            # print(self.mask[0])  # 5个1 5个1的来，因为有5个模态，时间序列长度为10
            # print(self.mask[0].shape)

    def forward(self, x, mask=None, CP_U=None, CP_V=None, CP_C=None):
        for layer_idx, (att_norm, self_att, cross_att, ff_norm, ff) in enumerate(self.layers):
            if mask is not None:
                x = x + drop_path(self_att(att_norm(x), mask, CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))  # Self Attention
                mha_out = cross_att(att_norm(x), att_norm(x), att_norm(x), mask, CP_U=CP_U, CP_V=CP_V,
                                    CP_C=CP_C)
                self.attention_output[layer_idx] = cross_att.att_weights
                x = x + drop_path(mha_out)
            elif self.mask is not None:
                # print("self.mask shape：", self.mask.shape)
                x = x + drop_path(self_att(att_norm(x), self.mask, CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))
                # print("after self attention x shape: ", x.shape)  (B, T*num_modalities"50, embed_size:384)
                mha_out = cross_att(att_norm(x), att_norm(x), att_norm(x), self.mask, CP_U=CP_U,
                                    CP_V=CP_V, CP_C=CP_C)
                # print("after cross attention x shape: ", mha_out.shape)  (B, T*num_modalities:50, embed_size:384)
                self.attention_output[layer_idx] = cross_att.att_weights
                x = x + drop_path(mha_out)
            else:  # no masking, just use full attention
                x = x + drop_path(self_att(att_norm(x), CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))
                mha_out = cross_att(att_norm(x), att_norm(x), att_norm(x), CP_U=CP_U, CP_V=CP_V, CP_C=CP_C)
                self.attention_output[layer_idx] = cross_att.att_weights
                x = x + drop_path(mha_out)

            # if not self.training:
            #     self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x), CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))
            # print("after ff_norm x shape: ", x.shape)  (1, T*num_modalities"50, embed_size:384)
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class TransformerDecoder_Skill_Prompt(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            num_heads,
            head_output_size,
            mlp_hidden_size,
            dropout,
            **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.attention_output = {}
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        Attention(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                        ),
                        Attention_Prompt(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                        ),
                        Norm(input_size),
                        TransformerFeedForwardNN(
                            input_size, mlp_hidden_size, dropout=dropout,
                        ),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

        self.prompt_num_elements = None
        self.prompt_mask = None

    def compute_mask(self, input_shape=None, prompt_shape=None):
        # print("input_shape", input_shape)  # torch.Size([B, 10, 5, 64])
        # print("prompt_shape", prompt_shape)  # [B, self.e_pool_size, 64]
        if (
                (self.num_elements is None)
                or (self.seq_len is None)
                or (self.num_elements != input_shape[2])
                or (self.seq_len != input_shape[1])
        ):
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

        if (
                (self.prompt_num_elements is None)
                or (self.prompt_num_elements != prompt_shape[1])
        ):
            self.prompt_num_elements = prompt_shape[1]
            self.prompt_original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.prompt_mask = 1 - self.prompt_original_mask.repeat_interleave(
                self.prompt_num_elements // self.seq_len, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)

    def forward(self, x, prompt, mask=None, prompt_mask=None):
        prompt_loss = torch.zeros((1,), requires_grad=True).cuda()
        for layer_idx, (att_norm, self_att, prompt_att, ff_norm, ff) in enumerate(self.layers):
            if self.training:
                p_list, loss, x = prompt.forward(x, layer_idx, x, train=True)
                prompt_loss += loss
            else:
                p_list, _, x = prompt.forward(x, layer_idx, x, train=False)
            if mask is not None and prompt_mask is not None:
                x = x + drop_path(self_att(att_norm(x), mask))  # Self Attention
                mha_out = prompt_att(att_norm(x), prompt_mask, prompt=p_list)
                self.attention_output[layer_idx] = prompt_att.att_weights
                x = x + drop_path(mha_out)
            elif self.mask is not None and self.prompt_mask is not None:
                # print("self.mask的维度：", self.mask.shape)  # (1, 50, 50)
                # print("self.prompt_mask的维度：", self.prompt_mask.shape)  # (1, 50, 100)
                x = x + drop_path(self_att(att_norm(x), self.mask))
                mha_out = prompt_att(att_norm(x), self.prompt_mask, prompt=p_list)
                self.attention_output[layer_idx] = prompt_att.att_weights
                x = x + drop_path(mha_out)
            else:  # no masking, just use full attention
                x = x + drop_path(self_att(att_norm(x)))
                mha_out = prompt_att(att_norm(x), prompt=p_list)
                self.attention_output[layer_idx] = prompt_att.att_weights
                x = x + drop_path(mha_out)

            # if not self.training:
            #     self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x, prompt_loss

    @property
    def device(self):
        return next(self.parameters()).device


class CP_TransformerDecoder_Skill_Prompt(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            num_heads,
            head_output_size,
            mlp_hidden_size,
            dropout,
            R,
            **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.attention_output = {}
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        Hier_CP_Attention(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                            R=R,
                        ),
                        CP_Attention_Skill_Prompt(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                            R=R,
                        ),
                        Norm(input_size),
                        CPTransformerFeedForwardNN(
                            input_size, mlp_hidden_size, dropout=dropout, R=R, num_heads=num_heads),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

        self.prompt_num_elements = None
        self.prompt_mask = None

    def compute_mask(self, input_shape, prompt_shape=None):
        if (
                (self.num_elements is None)
                or (self.seq_len is None)
                or (self.num_elements != input_shape[2])  # 5
                or (self.seq_len != input_shape[1])  # 10
        ):
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements
            # print(self.mask[0])  # 5个1 5个1的来，因为有5个模态，时间序列长度为10
            # print(self.mask[0].shape)

        if (
                (self.prompt_num_elements is None)
                or (self.prompt_num_elements != prompt_shape[1])
        ):
            self.prompt_num_elements = prompt_shape[1]
            self.prompt_original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.prompt_mask = 1 - self.prompt_original_mask.repeat_interleave(
                self.prompt_num_elements // self.seq_len, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)

    def forward(self, x, prompt, mask=None, prompt_mask=None, CP_U=None, CP_V=None, CP_C=None):
        prompt_loss = torch.zeros((1,), requires_grad=True).cuda()
        for layer_idx, (att_norm, self_cp_att, cp_prompt_att, ff_norm, ff) in enumerate(self.layers):
            if self.training:
                p_list, loss, x = prompt.forward(x, layer_idx, x, train=True)
                prompt_loss += loss
            else:
                p_list, _, x = prompt.forward(x, layer_idx, x, train=False)
            if mask is not None and prompt_mask is not None:
                x = x + drop_path(self_cp_att(att_norm(x), mask, CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))  # Self Attention
                mha_out = cp_prompt_att(att_norm(x), att_norm(x), att_norm(x), prompt_mask, CP_U=CP_U, CP_V=CP_V,
                                        CP_C=CP_C, prompt=p_list)
                self.attention_output[layer_idx] = cp_prompt_att.att_weights
                x = x + drop_path(mha_out)
            elif self.mask is not None and self.prompt_mask is not None:
                # print("self.mask shape：", self.mask.shape)
                # print("self.codebook_mask shape：", self.codebook_mask.shape)
                x = x + drop_path(self_cp_att(att_norm(x), self.mask, CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))
                # print("after self attention x shape: ", x.shape)  (B, T*num_modalities"50, embed_size:384)
                mha_out = cp_prompt_att(att_norm(x), att_norm(x), att_norm(x), self.prompt_mask, CP_U=CP_U,
                                        CP_V=CP_V, CP_C=CP_C, prompt=p_list)
                # print("after cross attention x shape: ", mha_out.shape)  (B, T*num_modalities:50, embed_size:384)
                self.attention_output[layer_idx] = cp_prompt_att.att_weights
                x = x + drop_path(mha_out)
            else:  # no masking, just use full attention
                x = x + drop_path(self_cp_att(att_norm(x), CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))
                mha_out = cp_prompt_att(att_norm(x), att_norm(x), att_norm(x), CP_U=CP_U, CP_V=CP_V,
                                        CP_C=CP_C, prompt=p_list)
                self.attention_output[layer_idx] = cp_prompt_att.att_weights
                x = x + drop_path(mha_out)

            # if not self.training:
            #     self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x), CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))
            # print("after ff_norm x shape: ", x.shape)  (1, T*num_modalities"50, embed_size:384)
        return x, prompt_loss

    @property
    def device(self):
        return next(self.parameters()).device


class TransformerDecoder_Action(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            num_heads,
            head_output_size,
            mlp_hidden_size,
            dropout,
            **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.attention_output = {}

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        Attention_Action(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                        ),
                        Norm(input_size),
                        TransformerFeedForwardNN(
                            input_size, mlp_hidden_size, dropout=dropout
                        ),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

    def compute_mask(self, input_shape):
        # input_shape = (:, seq_len, num_elements)
        if (
                (self.num_elements is None)
                or (self.seq_len is None)
                or (self.num_elements != input_shape[2])
                or (self.seq_len != input_shape[1])
        ):
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

    def forward(self, x, mask=None):
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            if mask is not None:
                x = x + drop_path(att(att_norm(x), mask))
            elif self.mask is not None:
                x = x + drop_path(att(att_norm(x), self.mask))
            else:  # no masking, just use full attention
                x = x + drop_path(att(att_norm(x)))

            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class CP_TransformerDecoder_Action(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            num_heads,
            head_output_size,
            mlp_hidden_size,
            dropout,
            R,
            **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.attention_output = {}

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        CP_Attention_Action(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                            R=R,
                        ),
                        Norm(input_size),
                        CPTransformerFeedForwardNN(
                            input_size, mlp_hidden_size, dropout=dropout, R=R, num_heads=num_heads),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

    def compute_mask(self, input_shape):
        # input_shape = (:, seq_len, num_elements)
        if (
                (self.num_elements is None)
                or (self.seq_len is None)
                or (self.num_elements != input_shape[2])
                or (self.seq_len != input_shape[1])
        ):
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

    def forward(self, x, mask=None, CP_U=None, CP_V=None, CP_C=None):
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            if mask is not None:
                x = x + drop_path(att(att_norm(x), mask, CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))
            elif self.mask is not None:
                x = x + drop_path(att(att_norm(x), self.mask, CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))
            else:  # no masking, just use full attention
                x = x + drop_path(att(att_norm(x), CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))

            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x), CP_U=CP_U, CP_V=CP_V, CP_C=CP_C))
        return x

    @property
    def device(self):
        return next(self.parameters()).device


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            input_size,
            num_layers,
            num_heads,
            head_output_size,
            mlp_hidden_size,
            dropout,
            **kwargs
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        self.drop_path = DropPath(dropout) if dropout > 0.0 else nn.Identity()

        self.attention_output = {}

        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        Norm(input_size),
                        Attention(
                            input_size,
                            num_heads=num_heads,
                            head_output_size=head_output_size,
                            dropout=dropout,
                        ),
                        Norm(input_size),
                        TransformerFeedForwardNN(
                            input_size, mlp_hidden_size, dropout=dropout
                        ),
                    ]
                )
            )

            self.attention_output[_] = None
        self.seq_len = None
        self.num_elements = None
        self.mask = None

    def compute_mask(self, input_shape):
        # input_shape = (:, seq_len, num_elements)
        if (
                (self.num_elements is None)
                or (self.seq_len is None)
                or (self.num_elements != input_shape[2])
                or (self.seq_len != input_shape[1])
        ):
            self.seq_len = input_shape[1]
            self.num_elements = input_shape[2]
            self.original_mask = (
                    torch.triu(torch.ones(self.seq_len, self.seq_len))
                    - torch.eye(self.seq_len, self.seq_len)
            ).to(self.device)
            self.mask = 1 - self.original_mask.repeat_interleave(
                self.num_elements, dim=-1
            ).repeat_interleave(self.num_elements, dim=-2).unsqueeze(0)
            # (1, N, N), N = seq_len * num_elements

    def forward(self, x, mask=None):
        # print("before attention shape:", x.shape)
        for layer_idx, (att_norm, att, ff_norm, ff) in enumerate(self.layers):
            if mask is not None:
                x = x + drop_path(att(att_norm(x), mask))
            elif self.mask is not None:
                x = x + drop_path(att(att_norm(x), self.mask))
            else:  # no masking, just use full attention
                x = x + drop_path(att(att_norm(x)))

            if not self.training:
                self.attention_output[layer_idx] = att.att_weights
            x = x + self.drop_path(ff(ff_norm(x)))
        # print("after attention shape:", x.shape)
        return x

    @property
    def device(self):
        return next(self.parameters()).device
