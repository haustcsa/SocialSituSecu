import torch.nn as nn
import torch
from transformers import BertModel
from fightingcv_attention.attention.SelfAttention import ScaledDotProductAttention
# 忽略 transformers 警告
from transformers import logging
logging.set_verbosity_error()

class CasRel(nn.Module):
    def __init__(self, config):
        super(CasRel, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        self.sub_heads_linear = nn.Linear(self.config.bert_dim, 1)
        self.sub_tails_linear = nn.Linear(self.config.bert_dim, 1)
        self.obj_heads_linear = nn.Linear(self.config.bert_dim, self.config.num_rel)
        self.obj_tails_linear = nn.Linear(self.config.bert_dim, self.config.num_rel)
        self.alpha = 0.25
        self.gamma = 2
        # 引入自注意力机制
        self.sa = ScaledDotProductAttention(d_model=768, d_k=768, d_v=768, h=8)
        self.sa1 = ScaledDotProductAttention(d_model=16, d_k=16, d_v=16, h=8)

   # bert编码
    def get_encoded_text(self, token_ids, mask):
        encoded_text = self.bert(token_ids, attention_mask=mask)[0]
        return encoded_text

    # 头实体识别 768降为1维
    def get_subs(self, encoded_text):
        pred_sub_heads = torch.sigmoid(self.sub_heads_linear(encoded_text))
        pred_sub_tails = torch.sigmoid(self.sub_tails_linear(encoded_text))
        return pred_sub_heads, pred_sub_tails

    # 特定关系-实体解码
    def get_objs_for_specific_sub(self, sub_head2tail, sub_len, encoded_text):
        sub = torch.matmul(sub_head2tail, encoded_text)  # batch size,1,dim
        sub_len = sub_len.unsqueeze(1)
        sub = sub / sub_len  # batch size, 1,dim
        encoded_text = encoded_text + sub
        # 自注意力机制
        encoded_text = self.sa(encoded_text, encoded_text, encoded_text)

        pred_obj_heads = torch.sigmoid(self.obj_heads_linear(encoded_text))
        pred_obj_tails = torch.sigmoid(self.obj_tails_linear(encoded_text))
        return pred_obj_heads, pred_obj_tails

    def forward(self, input_ids, mask, sub_head2tail, sub_len):
        """

        :param token_ids:[batch size, seq len]
        :param mask:[batch size, seq len]
        :param sub_head:[batch size, seq len]
        :param sub_tail:[batch size, seq len]
        :return:
        """
        # bert层得到的编码向量
        encoded_text = self.get_encoded_text(input_ids, mask)

        pred_sub_heads, pred_sub_tails = self.get_subs(encoded_text)

        sub_head2tail = sub_head2tail.unsqueeze(1)  # [[batch size,1, seq len]] (16,1,82)
        pred_obj_heads, pre_obj_tails = self.get_objs_for_specific_sub(sub_head2tail, sub_len, encoded_text)

        return {
            "pred_sub_heads": pred_sub_heads,
            "pred_sub_tails": pred_sub_tails,
            "pred_obj_heads": pred_obj_heads,
            "pred_obj_tails": pre_obj_tails,
            'mask': mask
        }

    def compute_loss(self, pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails, mask, sub_heads,
                     sub_tails, obj_heads, obj_tails):
        rel_count = obj_heads.shape[-1]
        rel_mask = mask.unsqueeze(-1).repeat(1, 1, rel_count)
        loss_1 = self.loss_fun(pred_sub_heads, sub_heads, mask)
        loss_2 = self.loss_fun(pred_sub_tails, sub_tails, mask)
        loss_3 = self.loss_fun(pred_obj_heads, obj_heads, rel_mask)
        loss_4 = self.loss_fun(pred_obj_tails, obj_tails, rel_mask)
        return loss_1 + loss_2 + loss_3 + loss_4

    def loss_fun(self, logist, label, mask):
        count = torch.sum(mask)
        logist = logist.view(-1)
        label = label.view(-1)
        mask = mask.view(-1)

        alpha_factor = torch.where(torch.eq(label, 1), 1 - self.alpha, self.alpha)
        focal_weight = torch.where(torch.eq(label, 1), 1 - logist, logist)

        loss = -(torch.log(logist) * label + torch.log(1 - logist) * (1 - label)) * mask
        return torch.sum(focal_weight * loss) / count

