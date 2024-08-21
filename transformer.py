import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import Resize, Compose, ToTensor
from torchvision.transforms.functional import to_pil_image

import matplotlib.pyplot as plt
from random import random

from einops.layers.torch import Rearrange
from einops import repeat

to_tensor = [Resize((144, 144)), ToTensor()]

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target=None):
        for t in self.transforms:
            image = t(image)
        return image, target
    
def show_images(images, num_samples=40, cols=8):
    plt.figure(figsize=(15, 15))
    index = int(len(dataset) / num_samples)

    print(images)

    for i, img in enumerate(images):
        if i % index == 0:
            plt.subplot(int(num_samples / cols) + 1, cols, int(i / index) + 1)
            plt.imshow(to_pil_image(img[0][0]))

dataset = OxfordIIITPet(root='.', download=True, transform=Compose(to_tensor))
#show_images(dataset)

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, emb_size=128) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.projection = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x)
    
sample_datapoint = torch.unsqueeze(dataset[0][0][0], 0)
print("Initial Shape: ", sample_datapoint.shape)

embedding = PatchEmbedding()(sample_datapoint)
print("Shape after Patch Embedding: ", embedding.shape)

class Head(nn.Module):
    def __init__(self, embed_dim: int, head_size: int, block_size: int, dropout=0.0, encoder=True) -> None:
        super().__init__()
    
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

        self.attention_map = None
        self.encoder = encoder

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = (q @ k.transpose(-2, -1)) * (C ** -0.5)
        
        if not self.encoder: 
            weights = weights.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        self.attention_map = weights
        out = weights @ v

        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, head_size: int, block_size: int, dropout=0.0, encoder=True) -> None:
        super().__init__()

        self.heads = nn.ModuleList([
            Head(embed_dim, head_size, block_size, dropout, encoder) for _ in range(n_heads)
        ])
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        scores = torch.cat([h(x) for h in self.heads], dim=-1)
        scores = self.projection(scores)
        scores = self.dropout(scores)

        return scores

print("Using Custom MultiHeadAttention Implementation: ", MultiHeadAttention(128, 8, 16, 5, 0)(torch.ones((1, 5, 128))).shape)

class FFD(nn.Module):
    def __init__(self, embed_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.ffd = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: Tensor) -> Tensor:
        return self.ffd(x)
    
class Block(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, block_size: int, dropout=0.0, encoder=True) -> None:
        self.sa = MultiHeadAttention(embed_dim, n_heads, embed_dim // n_heads, block_size, dropout, encoder)
        self.ffd = FFD(embed_dim, embed_dim * 4, dropout)

        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.sa(self.ln1(x)) + x
        x = self.ffd(self.ln2(x)) + x

        return x
    
class CustomViT(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, block_size: int, patch_size: int, n_layers: int, output_dim: int, input_dim: int, input_ch=3, dropout=0.0, encoder=True) -> None:
        super().__init__()
        
        self.num_patches = (input_dim // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(input_ch, patch_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        self.sa = nn.Sequential(
            *[Block(embed_dim, n_heads, block_size, dropout, encoder) for _ in range(n_layers)]
        )

        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )

    def forward(self, img: Tensor) -> Tensor:

        img_tokens = self.patch_embedding(img)
        B, T, C = img_tokens.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = B)
        img_tokens = torch.cat([cls_tokens, img_tokens], dim=1)
        position_embeddings = self.pos_embedding[:, :(T+1)]

        x = img_tokens + position_embeddings
        out = self.sa(x)
        out = self.cls_head(out[:, 0, :])

        return out

class Attention(nn.Module):
    def __init__(self, embd_dim, n_heads, dropout) -> None:
        super().__init__()
        self.n_heads = n_heads
        
        self.attention = nn.MultiheadAttention(embed_dim=embd_dim, num_heads=n_heads, dropout=dropout)

        self.q = torch.nn.Linear(embd_dim, embd_dim, bias=False)
        self.k = torch.nn.Linear(embd_dim, embd_dim, bias=False)
        self.v = torch.nn.Linear(embd_dim, embd_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attention_output, attention_weights = self.attention(q, k, v)

        return attention_output
    
print("Using MultiHead Implementation available in the library: ", Attention(embd_dim=128, n_heads=8, dropout=0)(torch.ones((1, 5, 128))).shape)

class PreNormalization(nn.Module):
    def __init__(self, net: nn.Module, embed_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.net = net

    def forward(self, x: Tensor) -> Tensor:
        return self.net(self.norm(x))
    
norm = PreNormalization(Attention(128, 8, 0), 128)
print(norm(torch.ones((1, 5, 128))).shape)

class Attention(nn.Module):
    def __init__(self, embd_dim, n_heads, dropout) -> None:
        super().__init__()
        self.n_heads = n_heads
        
        self.attention = nn.MultiheadAttention(embed_dim=embd_dim, num_heads=n_heads, dropout=dropout)

        self.q = torch.nn.Linear(embd_dim, embd_dim, bias=False)
        self.k = torch.nn.Linear(embd_dim, embd_dim, bias=False)
        self.v = torch.nn.Linear(embd_dim, embd_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attention_output, attention_weights = self.attention(q, k, v)

        return attention_output
    
print("Using MultiHead Implementation available in the library: ", Attention(embd_dim=128, n_heads=8, dropout=0)(torch.ones((1, 5, 128))).shape)

class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.ffd = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.ffd(x)
    
print("Feed Forward Network: ", FeedForward(128, 512, 0)(torch.ones((1, 5, 128))).shape)

class Residual(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + x
    
print("Residual Connection: ", Residual(Attention(128, 8, 0))(torch.ones((1, 5, 128))).shape)

class ViT(nn.Module):
    def __init__(self, in_channels=3, img_size=144, patch_size=4, embed_dim=32, n_layers=6, output_dim=37, dropout=0.1, heads=2):
        super(ViT, self).__init__()

        self.channels = in_channels
        self.height = img_size
        self.weight = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_layers = n_layers

        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim)) ## +1 for the cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.Sequential(
                Residual(PreNormalization(Attention(embed_dim, heads, dropout), embed_dim)),
                Residual(PreNormalization(FeedForward(embed_dim, embed_dim * 4, dropout), embed_dim))   
            ))

        self.dropout = nn.Dropout(dropout)
        
        self.cls_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim)
        )
    
    def forward(self, img: Tensor) -> Tensor:
        img_tokens = self.patch_embedding(img)
        
        B, T, C = img_tokens.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = B)
        img_tokens = torch.cat([cls_tokens, img_tokens], dim=1)

        position_embeddings = self.pos_embedding[:, :(T+1)]

        x = img_tokens + position_embeddings

        for i in range(self.n_layers):
            x = self.layers[i](x)

        out = self.cls_head(x[:,0,:])
        return out