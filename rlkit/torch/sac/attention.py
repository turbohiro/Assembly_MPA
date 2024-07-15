# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision.datasets import CIFAR100
from torchvision import transforms
import math

# PyTorch Lightning
import pytorch_lightning as pl


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=43):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TransformerFeature(pl.LightningModule):

    def __init__(self, input_dim, model_dim, num_heads, num_layers, max_iters, dropout=0.0, input_dropout=0.0):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        #super().__init__()
        #self.save_hyperparameters()
        #self._create_model()

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim)
        )
        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(num_layers=self.hparams.num_layers,
                                              input_dim=self.hparams.model_dim,
                                              dim_feedforward=2*self.hparams.model_dim,
                                              num_heads=self.hparams.num_heads,
                                              dropout=self.hparams.dropout)
        # Output classifier per sequence lement
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        #x = self.output_net(x)
        return x


encoder_layer = nn.TransformerEncoderLayer(
    d_model=64, 
    nhead=2, 
    dim_feedforward=64, 
    dropout=0.0)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
embedding = nn.Embedding(43, 64)

#out = torch.rand(5,64).unsqueeze(dim=-1).permute(0, 2, 1)
#out = out.expand(-1,43,-1)
batch = 2
out = torch.rand(batch,43).unsqueeze(dim=-1).permute(0, 2, 1)
pos = F.one_hot(torch.arange(0,43),43).expand(batch,-1,-1)
out = torch.cat((pos,out),dim=1)
m = nn.Conv1d(44,64,1)(out).permute(0,2,1)
out = transformer_encoder(m)
h_final = nn.Flatten()(out)
#pos_embed = nn.Parameter(torch.zeros(1, 43 + 1, 64))
#pos = nn.Parameter(torch.zeros(1,43)).expand(batch,-1)
import pdb
pdb.set_trace()
out = out + pos



cls_token = nn.Parameter(torch.zeros(1, 1, 64))
out = torch.cat((cls_token,out),dim=1)
feature1 = torch.cat((out,pos),dim=1)
import pdb
pdb.set_trace()
#pos_embed = nn.Parameter(torch.zeros(5, 44, 64))
# out = embedding(out) * math.sqrt(64)
a = PositionalEncoding(64)
b = a(out)
#out = out.permute(2, 0, 1)
#print('transpose (n_length, n_samples, n_channel)', out.shape)

#out = transformer_encoder(out)
#print('transformer_encoder', out.shape)
## learnable positional encoding
#lang_emb_dim, lang_max_seq_len = 512, 64  
#pos_encoding = nn.Parameter(torch.randn(1,lang_max_seq_len,64))
#x = x+pos_encoding
#inp_data = F.one_hot(x, num_classes=1).float()
#
#feature = TransformerFeature(x,input_dim=64,model_dim=32,num_heads=1,num_layers=1,dropout=0.0)

# import torch
# import torch.nn as nn


# class PatchEmbedding(nn.Module):
#     """
#     Patch Embedding & Position Embedding & Class Token
#     """
#     def __init__(self, in_channels, img_size, patch_size, embed_dim=768):
#         super(PatchEmbedding, self).__init__()
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = (img_size[0] // patch_size[0]) \
#                            * (img_size[1] // patch_size[1])
#         self.patchEmbedding = nn.Conv2d(
#             in_channels,
#             embed_dim,
#             kernel_size=patch_size,
#             stride=patch_size,
#         )
#         self.pos_embed = nn.Parameter(
#             torch.zeros(1, self.num_patches + 1, embed_dim)
#         )
#         self.cls_token = nn.Parameter(
#             torch.zeros(1, 1, embed_dim)
#         )

#     def forward(self, x):
#         B, C, H, W = x.shape
#         assert H == self.img_size[0] and W == self.img_size[1], \
#             f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         # Patch Embedding
#         x = self.patchEmbedding(x)
#         import pdb
#         pdb.set_trace()
#         x = x.flatten(2).transpose(1, 2)  # .transpose(1, 2) == .permute(0, 2, 1)
#         # Class Token
#         cls_token = self.cls_token.expand(B, -1, -1)  # 扩展B维度，其他维度不变
#         x = torch.cat((cls_token, x), dim=1)
#         # Position Embedding
#         pos_embed = self.pos_embed.expand(B, -1, -1)
#         return x + pos_embed


# def main():
#     img_size = (224, 224)
#     patch_size = (16, 16)
#     embed_dim = 768
#     x = torch.randn(2, 3, 224, 224)
#     print(f"input shape is {x.shape}")
#     model = PatchEmbedding(3, img_size, patch_size, embed_dim)
#     out = model(x)
#     print(f"output shape is {out.shape}")
#     import pdb
#     pdb.set_trace()


# if __name__ == "__main__":
#     main()
    
import pdb
pdb.set_trace()

# import torch
# from h_transformer_1d import HTransformer1D

# model = HTransformer1D(
#     num_tokens = 256,          # number of tokens
#     dim = 512,                 # dimension
#     depth = 12,                # depth
#     causal = False,            # autoregressive or not
#     max_seq_len = 8192,        # maximum sequence length
#     heads = 8,                 # heads
#     dim_head = 64,             # dimension per head
#     block_size = 128,          # block size
#     reversible = True,         # use reversibility, to save on memory with increased depth
#     shift_tokens = True        # whether to shift half the feature space by one along the sequence dimension, for faster convergence (experimental feature)
# )

# x = torch.randint(0, 256, (1, 8000))   # variable sequence length
# #mask = torch.ones((1, 8000)).bool()    # variable mask length
# mask=None
# # network will automatically pad to power of 2, do hierarchical attention, etc

# logits = model(x, mask = mask) # (1, 8000, 256)
# import pdb
# pdb.set_trace()
