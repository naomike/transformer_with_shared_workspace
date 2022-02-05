import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops.layers.torch import Rearrange


class SharedWorkspace(nn.Module):
    def __init__(self, num_patches, dim, broadcast=True):
        super().__init__()

        self.num_patches = num_patches
        self.broadcast = broadcast

        self.attention_write = nn.MultiheadAttention(dim, 4)
        self.attention_broadcast = nn.MultiheadAttention(dim, 4)
        self.W1 = nn.Parameter(torch.Tensor(np.random.normal(size=(dim, dim))))
        self.Wi_linear = nn.Linear(dim, dim)
        self.Wf_linear = nn.Linear(dim, dim)


    def forward(self, Mt, x):
        # write into the shared workspace
        M = self.attention_write(
            Mt.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2)
        )[0].permute(1, 0, 2)

        _x = 0
        for i in range(self.num_patches):
            _x += F.relu(torch.matmul(x[:, i, :], self.W1))
        _x /= self.num_patches
        _x = torch.unsqueeze(_x, 1)
        K = _x + torch.tanh(Mt)
        _I = torch.sigmoid(self.Wi_linear(K))
        _F = torch.sigmoid(self.Wf_linear(K))
        Mt = torch.mul(_I, torch.tanh(M)) + torch.mul(_F, Mt)

        # broadcast from the shared workspace
        x = x + self.attention_broadcast(
            x.permute(1, 0, 2), Mt.permute(1, 0, 2), Mt.permute(1, 0, 2)
        )[0].permute(1, 0, 2)

        return Mt, x


class EasyTransformer(nn.Module):
    def __init__(self, device, with_attention, with_shared_workspace, img_size, \
        patch_size=4, channels=3, dim=128):
        super().__init__()

        image_height, image_width = img_size, img_size
        patch_height, patch_width = patch_size, patch_size

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        self.dim = dim
        self.num_patches = num_patches

        self.device = device
        self.with_attention = with_attention
        self.with_shared_workspace = with_shared_workspace

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.dropout = nn.Dropout(0.25)
        self.layer_norm = nn.LayerNorm(dim)
        self.ff_layers = nn.ModuleList([])

        if with_attention:
            for _ in range(3):
                self.ff_layers.append(nn.ModuleList([
                    nn.MultiheadAttention(dim, 4),
                    nn.Sequential(
                        nn.Linear(dim, 64),
                        nn.GELU(),
                        nn.Linear(64, dim),
                    )
                ]))
        else:
            for _ in range(3):
                self.ff_layers.append(nn.Sequential(
                    nn.Linear(dim, 64),
                    nn.GELU(),
                    nn.Linear(64, dim),
                ))

        self.shared_workspace = SharedWorkspace(num_patches, dim)
        self.mlp_head1 = nn.Linear(num_patches*dim, dim)
        self.mlp_head2 = nn.Linear(dim, 2)


    def forward(self, x):
        x = self.to_patch_embedding(x)
        x += self.pos_embedding
        x = self.dropout(x)

        if self.with_attention:
            for attn, ff in self.ff_layers:
                x = x + self.layer_norm(attn(
                    x.permute(1, 0, 2), x.permute(1, 0, 2), x.permute(1, 0, 2)
                )[0].permute(1, 0, 2))
                x = x + self.layer_norm(ff(x))
        elif self.with_shared_workspace:
            Mt = torch.Tensor(np.random.normal(size=(x.size()[0], 8, self.dim))).to(self.device)
            for ff in self.ff_layers:
                Mt, x = self.shared_workspace(Mt, x)
                x = x + self.layer_norm(ff(x))
        else:
            for ff in self.ff_layers:
                x = x + self.layer_norm(ff(x))

        x = x.view(-1, self.num_patches*self.dim)
        x = F.relu(self.mlp_head1(x))
        y = self.mlp_head2(x)

        return y
