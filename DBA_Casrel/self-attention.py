import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps


    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(SelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 设超参数num_attention_heads为自注意力机制的头数
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        # 定义Wq,Wk,Wv三个矩阵
        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        # 切分为num_attention_heads个头，并变换维度
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        # 计算过程中张量维度的变化 维度变化：input_tensor---mixed_query_layer
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        # 维度变化：mixed_query_layer---query_layer
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # 矩阵Q和K相乘，得到注意力矩阵，并除以向量的维度的开方，防止注意力分数随维度增大而增大
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 注意力矩阵归一化 维度变化：attention_scores---attention_probs
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        # 将注意力矩阵乘以矩阵V attention_probs*value_layer---context_layer
        context_layer = torch.matmul(attention_probs, value_layer)

        # 变换context_layer维度，为了后面将各头得到的结果拼接
        # 这里contiguous()是将tensor的内存变成连续的，为后面的view()做准备
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # 将各头的结果拼接起来
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states