import os
import re
import torchvision
import timm
import torch
from torch import nn
from typing import List
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..', '..'))


def _load_xrv_densenet121(pretrained):
    import torchxrayvision as xrv
    model = xrv.models.DenseNet(weights="densenet121-res224-all" if pretrained else None)
    # XRV conv0 expects 1-channel grayscale; replace with 3-channel RGB conv.
    # Initialize by averaging the pretrained 1-ch kernel across 3 input channels so
    # that feeding (R+G+B)/3 ≈ grayscale reproduces the original pretrained response.
    old_conv = model.features.conv0
    new_conv = nn.Conv2d(3, old_conv.out_channels,
                         kernel_size=old_conv.kernel_size,
                         stride=old_conv.stride,
                         padding=old_conv.padding,
                         bias=False)
    if pretrained:
        new_conv.weight.data = old_conv.weight.data.repeat(1, 3, 1, 1) / 3.0
    model.features.conv0 = new_conv
    return model


class _CheXFoundFeatures(nn.Module):
    """Returns spatial patch-token feature maps [B, 1024, H//16, W//16] from CheXFound ViT-L."""
    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x):
        out = self.vit.get_intermediate_layers(x, n=1, reshape=True, return_class_token=False)
        return out[0]


class _CheXFoundWrapper(nn.Module):
    """Minimal wrapper with named children 'features'/'classifier' for IntermediateLayerExtracter."""
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.classifier = nn.Identity()

    def forward(self, x):
        return self.features(x)


def _load_chexfound(pretrained):
    import sys
    chexfound_lib = os.path.join(_PROJECT_ROOT, 'lib', 'chexfound')
    if chexfound_lib not in sys.path:
        sys.path.insert(0, chexfound_lib)
    from chexfound.models.vision_transformer import vit_large

    # Build ViT-L/16 at img_size=512 to match checkpoint pos_embed [1, 1025, 1024].
    # At inference, get_intermediate_layers calls interpolate_pos_encoding for other sizes.
    vit = vit_large(
        img_size=512, patch_size=16, ffn_layer='swiglufused', block_chunks=0,
        qkv_bias=True, proj_bias=True, ffn_bias=True, num_register_tokens=4,
        interpolate_antialias=False, interpolate_offset=0.1,
        init_values=1e-5, drop_path_rate=0.0,
    )

    # xFormers memory_efficient_attention doesn't support SM 12.0 (Blackwell) GPUs.
    # Disable it only for attention; SwiGLUFFNFused keeps its xFormers path.
    import chexfound.layers.attention as _cf_attn
    _cf_attn.XFORMERS_AVAILABLE = False

    if pretrained:
        ckpt_path = os.path.join(_PROJECT_ROOT, 'pretrained', 'chexfound',
                                 'CheXFound', 'teacher_checkpoint.pth')
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # Checkpoint blocks are chunked: blocks.{chunk}.{global_block}.param
        # Model (block_chunks=0) expects: blocks.{global_block}.param
        def _remap(k):
            m = re.match(r'^blocks\.(\d+)\.(\d+)\.(.*)', k)
            return f'blocks.{m.group(2)}.{m.group(3)}' if m else k
        backbone_sd = {
            _remap(k.replace('backbone.', '')): v
            for k, v in ckpt['teacher'].items() if k.startswith('backbone.')
        }
        missing, unexpected = vit.load_state_dict(backbone_sd, strict=False)
        assert not unexpected, f'Unexpected keys in CheXFound checkpoint: {unexpected[:5]}'

    # Gradient checkpointing every other block: recomputes 12 of 24 blocks during backward.
    # Halves recompute overhead vs all-blocks while still cutting activation memory ~4×.
    from torch.utils.checkpoint import checkpoint as _ckpt
    for i, blk in enumerate(vit.blocks):
        if i % 2 == 0:
            _orig = blk.forward
            blk.forward = lambda x, _f=_orig: _ckpt(_f, x, use_reentrant=False)

    return _CheXFoundWrapper(_CheXFoundFeatures(vit))


_MODELS = {
    "resnet18": torchvision.models.resnet.resnet18,
    "resnet34": torchvision.models.resnet.resnet34,
    "resnet50": torchvision.models.resnet.resnet50,
    "resnet101": torchvision.models.resnet.resnet101,
    "resnet152": torchvision.models.resnet.resnet152,
    "tresnet_m": timm.models.tresnet.tresnet_m,
    "tresnet_l": timm.models.tresnet.tresnet_l,
    "tresnet_xl": timm.models.tresnet.tresnet_xl,
    "tresnet_v2_l": timm.models.tresnet.tresnet_v2_l,
    "convnext_base":  lambda pretrained: timm.create_model("convnext_base.fb_in22k_ft_in1k",  pretrained=pretrained),
    "convnext_large": lambda pretrained: timm.create_model("convnext_large.fb_in22k_ft_in1k", pretrained=pretrained),
    "densenet121_cxr": _load_xrv_densenet121,
    "chexfound":       _load_chexfound,
}


@dataclass
class ModelInfo:
    layer_featuremap: str
    layer_fc: str
    dim_featuremap: int
    dim_fc: int
    dim_out: int
    downsample_ratio: int


_MODELS_INFO = {
    "resnet18": ModelInfo("layer4", "fc", 512, 512, 1000, 32),
    "resnet34": ModelInfo("layer4", "fc", 512, 512, 1000, 32),
    "resnet50": ModelInfo("layer4", "fc", 2048, 2048, 1000, 32),
    "resnet101": ModelInfo("layer4", "fc", 2048, 2048, 1000, 32),
    "resnet152": ModelInfo("layer4", "fc", 2048, 2048, 1000, 32),
    "tresnet_m": ModelInfo("body", "head.fc", 2048, 2048, 1000, 32),
    "tresnet_l": ModelInfo("body", "head.fc", 2432, 2432, 1000, 32),
    "tresnet_xl": ModelInfo("body", "head.fc", 2656, 2656, 1000, 32),
    "tresnet_v2_l": ModelInfo("body", "head.fc", 2048, 2048, 1000, 32),
    "convnext_base":  ModelInfo("norm_pre", "head.fc", 1024, 1024, 1000, 32),
    "convnext_large": ModelInfo("norm_pre", "head.fc", 1536, 1536, 1000, 32),
    "densenet121_cxr": ModelInfo("features", "classifier", 1024, 1024, 18, 32),
    "chexfound":       ModelInfo("features", "classifier", 1024, 1024,  0, 16),
}


def available_models() -> List[str]:
    """Returns the names of available CNN models"""
    return list(_MODELS.keys())


def create_cnn_backbone(name, *, num_classes=None, pretrained=False):
    assert name in _MODELS.keys()
    if "tresnet" in name:
        model = _MODELS[name](pretrained=pretrained)

    elif "resnet" in name:
        model = _MODELS[name](weights="DEFAULT" if pretrained else None)

    elif "convnext" in name:
        model = _MODELS[name](pretrained)

    elif name in ("densenet121_cxr", "chexfound"):
        model = _MODELS[name](pretrained)

    if num_classes is not None:
        info = _MODELS_INFO[name]
        setattr(model, info.layer_fc, nn.Linear(info.dim_fc, num_classes, bias=True))

    return model


class IntermediateLayerExtracter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layer: str) -> None:
        if return_layer not in [name for name, _ in model.named_children()]:
            raise ValueError("return_layer are not present in model")
        layers = dict()
        for name, module in model.named_children():
            layers[name] = module
            if name == return_layer:
                break

        super().__init__(layers)
        self.return_layer = return_layer

    def forward(self, x):
        for name, module in self.items():
            x = module(x)
        return x


def create_featuremap_backbone(name, *, pretrained=False):
    model = create_cnn_backbone(name, pretrained=pretrained)
    info = _MODELS_INFO[name]
    model = IntermediateLayerExtracter(model, info.layer_featuremap)
    return model


def _model_info(name, model):
    if "tresnet" in name:
        info = {
            "layer_featuremap": "layer4",
            "layer_fc": "fc",
            "dim_featuremap": model.layer4[-1].conv1.in_channels,
            "dim_fc": model.fc.in_features,
            "dim_out": model.fc.out_features,
            "downsample_ratio": 32,
        }

    elif "resnet" in name:
        info = {
            "layer_featuremap": "body",
            "layer_fc": "head.fc",
            "dim_featuremap": model.body[-1][-1].conv1.in_channels,
            "dim_fc": model.head.fc.in_features,
            "dim_out": model.head.fc.out_features,
            "downsample_ratio": 32,
        }

    return info
