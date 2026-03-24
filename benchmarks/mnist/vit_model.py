"""
Vision Transformer (ViT) for MNIST Classification
===================================================
A minimal but complete ViT implementation.

Architecture overview:
  - Patch Embedding: splits each 28x28 image into 7x7 patches (patch_size=4 → 49 patches)
  - Positional Encoding: learnable 1D embeddings added to patch tokens + class token
  - Transformer Encoder: N layers of Multi-Head Self-Attention + FFN
  - MLP Head: classifies the [CLS] token into 10 digit classes

Usage:
    python vit_mnist.py
"""

import mytorch
import mytorch.nn as nn


# ── Hyper-parameters ────────────────────────────────────────────────────────
IMAGE_SIZE   = 28          # MNIST images are 28×28
PATCH_SIZE   = 4           # Each patch is 4×4 pixels  →  (28/4)² = 49 patches
IN_CHANNELS  = 1           # Grayscale
NUM_CLASSES  = 10

DIM          = 64          # Token / embedding dimension
DEPTH        = 6           # Number of Transformer encoder layers
NUM_HEADS    = 8           # Attention heads
MLP_DIM      = 128         # Hidden size inside the feed-forward block
DROPOUT      = 0.1

DEVICE       = "gpu:0"


# ── Building blocks ──────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Splits the image into non-overlapping patches and projects each
    flattened patch to `dim`-dimensional space using a Conv2d layer
    (equivalent to a linear projection but more efficient).
    """
    def __init__(self, image_size, patch_size, in_channels, dim):
        super().__init__()
        assert image_size % patch_size == 0, "Image size must be divisible by patch size."
        self.num_patches = (image_size // patch_size) ** 2
        # A single Conv2d with kernel=stride=patch_size extracts & projects patches
        self.projection = nn.Conv2d(in_channels, dim,
                                    kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.projection(x)          # (B, dim, H/P, W/P)
        x = x.flatten(2)                # (B, dim, num_patches)
        x = x.transpose(1, 2)          # (B, num_patches, dim)
        return x


class MultiHeadSelfAttention(nn.Module):
    """Standard scaled dot-product multi-head self-attention."""
    def __init__(self, dim, num_heads, dropout):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv   = nn.Linear(dim, dim * 3, bias=False)
        self.proj  = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        nn.init.trunc_normal_(self.qkv.weight, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, N, C = x.shape
        # Compute Q, K, V in one matrix multiply then split
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)   # (3, B, heads, N, head_dim)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network with GELU activation."""
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential([
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        ])

        for m in self.net._modules.values():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    """One Transformer encoder layer: LayerNorm → Attention → residual,
       then LayerNorm → FFN → residual (Pre-LN formulation)."""
    def __init__(self, dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ── Vision Transformer ───────────────────────────────────────────────────────

class VisionTransformer(nn.Module):
    """
    Full ViT model:
      1. Embed patches
      2. Prepend learnable [CLS] token
      3. Add learnable positional embeddings
      4. Pass through Transformer encoder stack
      5. Classify via the [CLS] token representation
    """
    def __init__(self, image_size, patch_size, in_channels, num_classes,
                 dim, depth, num_heads, mlp_dim, dropout=0.):
        super().__init__()

        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, dim)
        self.dim = dim
        num_patches = self.patch_embed.num_patches

        # Learnable [CLS] token and positional embeddings
        self.cls_token = mytorch.zeros([1, 1, dim], True)
        self.pos_embed = mytorch.zeros([1, num_patches + 1, dim], True)
        self.pos_drop  = nn.Dropout(dropout)

        # Transformer encoder stack
        self.encoder = nn.Sequential(
            [TransformerEncoderBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

        # Classification head
        self.head = nn.Linear(dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # 1. Patch embedding: (B, num_patches, dim)
        x = self.patch_embed(x)

        # 2. Prepend [CLS] token
        cls = self.cls_token.broadcast_to([B, 1, self.dim])   # (B, 1, dim)
        x   = mytorch.concatenate([cls, x], dim=1)          # (B, num_patches+1, dim)

        # 3. Add positional embeddings
        x = self.pos_drop(x + self.pos_embed)

        # 4. Transformer encoder
        x = self.encoder(x)
        x = self.norm(x)

        # 5. Classify from [CLS] token
        cls_out = x[:, 0]                          # (B, dim)
        return self.head(cls_out)                  # (B, num_classes)



def create_vit():
    return VisionTransformer(
        image_size  = IMAGE_SIZE,
        patch_size  = PATCH_SIZE,
        in_channels = IN_CHANNELS,
        num_classes = NUM_CLASSES,
        dim         = DIM,
        depth       = DEPTH,
        num_heads   = NUM_HEADS,
        mlp_dim     = MLP_DIM,
        dropout     = DROPOUT
    ).to(DEVICE)