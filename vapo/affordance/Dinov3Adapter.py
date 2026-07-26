from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DINOv3PyramidEncoder(nn.Module):
    """
    Adapta um DINOv3 ViT para o formato de encoder esperado pelo SMP.

    Retorna:
        [
            x,       # H
            feat_1,  # H/2
            feat_2,  # H/4
            feat_3,  # H/8
            feat_4,  # H/16
        ]
    """

    def __init__(
        self,
        backbone: nn.Module,
        out_channels,
        depth: int,
        in_channels: int = 1,
        freeze_backbone: bool = True,
        normalize_input: bool = True,
    ):
        super().__init__()

        if len(out_channels) != depth + 1:
            raise ValueError(
                f"Esperava {depth + 1} valores em out_channels, "
                f"mas recebeu {len(out_channels)}."
            )

        self.backbone = backbone
        self._out_channels = tuple(out_channels)
        self._depth = depth
        self._in_channels = in_channels

        self.freeze_backbone = freeze_backbone
        self.normalize_input = normalize_input

        embed_dim = backbone.embed_dim
        n_blocks = backbone.n_blocks

        # ViT-S/B: depth=4 -> [2, 5, 8, 11]
        # depth=3 -> [3, 7, 11]
        self.layer_indices = [
            ((i + 1) * n_blocks) // depth - 1
            for i in range(depth)
        ]

        # Uma projeção para cada nível da pirâmide.
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(embed_dim, channels, kernel_size=1, bias=False),
                    nn.GroupNorm(_group_count(channels), channels),
                    nn.GELU(),
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(channels), channels),
                    nn.GELU(),
                )
                for channels in self._out_channels[1:]
            ]
        )

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    @property
    def out_channels(self):
        return self._out_channels[: self._depth + 1]

    @property
    def output_stride(self):
        return 2 ** self._depth

    def train(self, mode: bool = True):
        super().train(mode)

        # Mantém LayerNorm/dropout do backbone em modo de inferência.
        if self.freeze_backbone:
            self.backbone.eval()

        return self

    def _prepare_input(self, x):
        if x.shape[1] == 1:
            x_rgb = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 3:
            x_rgb = x
        else:
            raise ValueError(
                "O DINOv3 espera entrada RGB. "
                f"Recebido tensor com {x.shape[1]} canais."
            )

        if self.normalize_input:
            mean = self.mean.to(dtype=x_rgb.dtype)
            std = self.std.to(dtype=x_rgb.dtype)
            x_rgb = (x_rgb - mean) / std

        return x_rgb

    def forward(self, x):
        _, _, height, width = x.shape

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"O tamanho da imagem deve ser divisível por 16. "
                f"Recebido: {height}x{width}."
            )

        dino_input = self._prepare_input(x)

        context = torch.no_grad() if self.freeze_backbone else nullcontext()

        with context:
            dino_features = self.backbone.get_intermediate_layers(
                dino_input,
                n=self.layer_indices,
                reshape=True,
                return_class_token=False,
            )

        pyramid = [x]

        for level, (feature, projection) in enumerate(
            zip(dino_features, self.projections),
            start=1,
        ):
            feature = projection(feature)

            target_size = (
                height // (2 ** level),
                width // (2 ** level),
            )

            feature = F.interpolate(
                feature,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

            pyramid.append(feature)

        return pyramid