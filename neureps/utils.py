import torch
import torch.nn as nn

from neureps import modules


def spatial_unsqueeze(x, num_dims: int):
    non_feature_dims = x.ndim - 1
    return x[(slice(None),) * non_feature_dims + (None,) * num_dims + (slice(None),)]


def meshgrid(*size: int):
    tensors = [torch.linspace(-1, 1, n) for n in size]

    return torch.stack(
        torch.meshgrid(tensors, indexing='ij'), -1
    ).unsqueeze(0)


def get_block_factory(activation='siren', bias=True, **kwargs):
    activations = {
        'siren': (modules.SirenBlockFactory, dict(
            linear_cls=nn.Linear, bias=bias
        )),
        'relu': (modules.LinearBlockFactory, dict(
            linear_cls=nn.Linear, activation_cls=nn.ReLU, bias=bias
        )),
        'leaky_relu': (modules.LinearBlockFactory, dict(
            linear_cls=nn.Linear, activation_cls=lambda: nn.LeakyReLU(0.2), bias=bias
        )),
        'swish': (modules.LinearBlockFactory, dict(
            linear_cls=nn.Linear, activation_cls=modules.Swish, bias=bias
        ))
    }

    if activation not in activations:
        raise ValueError(f'Unknown activation {activation}. Available options are {list(activations)}')

    block_factory, default_kwargs = activations[activation]
    return block_factory(**{**default_kwargs, **kwargs})


def psnr(x):
    return -10 * torch.log10(x)
