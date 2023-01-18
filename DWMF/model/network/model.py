import torch
import torch.nn as nn
from transformers import BertModel
from typing import Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=40,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return x


class ModalityFusion(torch.nn.Module):
    def __init__(self):
        super(ModalityFusion, self).__init__()

        bert_feature_size = 768
        vit_feature_size = 768
        lstm_feature_size = 128

        self.bert = torch.nn.Linear(bert_feature_size, 128)
        self.vit = torch.nn.Linear(vit_feature_size, 128)

        self.bert_weight = torch.nn.Linear(bert_feature_size, 1)
        self.vit_weight = torch.nn.Linear(vit_feature_size, 1)
        self.lstm_weight = torch.nn.Linear(lstm_feature_size, 1)

        self.fc = nn.Linear(128, 2)

    def forward(self, bert_feature, vit_feature, lstm_feature):
        bert_vector = torch.tanh(self.bert(bert_feature))
        vit_vector = torch.tanh(self.vit(vit_feature))

        bert_score = self.bert_weight(bert_feature)
        lstm_score = self.lstm_weight(lstm_feature)
        vit_score = self.vit_weight(vit_feature)

        score = torch.softmax(torch.hstack([bert_score, vit_score, lstm_score]), dim=1)
        output = score[:, 0:1] * bert_vector + \
                 score[:, 1:2] * vit_vector + \
                 score[:, 2:] * lstm_feature
        output = torch.log_softmax(self.fc(output), dim=1)
        return output


class LayerNormal(nn.Module):
    def __init__(self, hidden_size, esp=1e-6):
        super(LayerNormal, self).__init__()
        self.esp = esp
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mu = torch.mean(input=x, dim=-1, keepdim=True)
        sigma = torch.std(input=x, dim=-1, keepdim=True).clamp(min=self.esp)
        out = (x - mu) / sigma
        out = out * self.weight.expand_as(out) + self.bias.expand_as(out)
        return out


class LsTModel(nn.Module):
    def __init__(self,
                 input_size: Optional[int] = 100,
                 hidden_size: Optional[int] = 128,
                 num_layers: Optional[int] = 1
                 ):
        super(LsTModel, self).__init__()

        self.sen_rnn = nn.LSTM(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               bidirectional=False)

        self.LayerNormal = LayerNormal(hidden_size)
        self.fc = nn.Linear(128, 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.normal_(0.0, 0.001)

    def forward(self, x):

        x, _ = self.sen_rnn(x, None)
        x = self.LayerNormal(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        x = torch.log_softmax(x, dim=1)
        return x


class BertClassModel(nn.Module):

    def __init__(self, bert_path):
        super(BertClassModel, self).__init__()
        self.bertModel = BertModel.from_pretrained(bert_path)
        self.fc = nn.Linear(768, 2)

    def forward(self, bert_ids, bert_mask):
        x = self.bertModel(bert_ids, token_type_ids=None, attention_mask=bert_mask)
        x = x.last_hidden_state
        x = torch.mean(x, dim=1)
        x = torch.log_softmax(self.fc(x), dim=1)
        return x


class VitModel(nn.Module):
    def __init__(self):
        super(VitModel, self).__init__()
        self.vit = ViT(image_size=128, patch_size=16, num_classes=None, dim=768, depth=1, channels=3,
                       heads=16, mlp_dim=2048, dropout=0.5, emb_dropout=0.5
                       )
        self.fc = nn.Linear(768, 2)

    def forward(self, img_array):
        x = self.vit(img_array)
        x = torch.log_softmax(self.fc(x), dim=1)
        return x


class FusionModel(nn.Module):
    def __init__(self, bert_path):
        super(FusionModel, self).__init__()
        self.bertModel = BertModel.from_pretrained(bert_path)
        self.vit = ViT(image_size=128, patch_size=16, num_classes=None, dim=768, depth=1, channels=3,
                       heads=16, mlp_dim=2048, dropout=0.5, emb_dropout=0.5
                       )
        self.lstm = nn.LSTM(input_size=100,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.fusion = ModalityFusion()

    def forward(self, bert_ids, bert_mask, img_array, lstm_array):
        x = self.bertModel(bert_ids, token_type_ids=None, attention_mask=bert_mask)
        bert_x = x.last_hidden_state
        bert_x = torch.tanh(torch.mean(bert_x, dim=1))

        img_x = torch.tanh(self.vit(img_array))

        lstm_x, _ = self.lstm(lstm_array, None)
        lstm_x = torch.tan(torch.mean(lstm_x, dim=1))
        fusion = self.fusion(bert_x, img_x, lstm_x)

        return fusion


def get_parameter_number(net, name):
    total_num = sum(p.numel() for p in net.parameters())
    return {'name{}: ->:{}'.format(name, total_num)}


if __name__ == '__main__':
    bert_path_ = r'../bert-base-chinese'
    bert_ids_ = torch.tensor([[345, 232, 13, 544, 2323]])
    bert_mask_ = torch.tensor([[1, 1, 0, 0, 0]])
    img_data_ = torch.rand(1, 3, 128, 128)
    lstm_data_ = torch.rand(1, 64, 100)

    bert_model = BertClassModel(bert_path_)
    bert_out = bert_model(bert_ids_, bert_mask_)
    print(bert_out.shape)

    vit_model = VitModel()
    vit_out = vit_model(img_data_)
    print(vit_out.shape)

    lstm_model = LsTModel()
    lstm_out = lstm_model(lstm_data_)
    print(lstm_out.shape)

    fusion_model = FusionModel(bert_path_)
    fusion_out = fusion_model(bert_ids_, bert_mask_, img_data_, lstm_data_)
    print(fusion_out.shape)



