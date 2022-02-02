import math
import torch
import torch.nn as nn

from copy import copy
from functools import partial
from typing import Optional, List, Callable
from math import sqrt
from neureps import utils, encodings


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

    def __repr__(self):
        return f'Sine(w0={self.w0})'


class LinearBlock(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 linear_cls,
                 activation=nn.ReLU,
                 bias=True,
                 is_first=False,
                 is_last=False):
        super().__init__()
        self.in_f = in_features
        self.out_f = out_features
        self.linear = linear_cls(in_features, out_features, bias=bias)
        self.bias = bias
        self.is_first = is_first
        self.is_last = is_last
        self.activation = None if is_last else activation()

    def forward(self, x):
        x = self.linear(x)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x

    def __repr__(self):
        return f'LinearBlock(in_features={self.in_f}, out_features={self.out_f}, linear_cls={self.linear}, ' \
               f'activation={self.activation}, bias={self.bias}, is_first={self.is_first}, is_last={self.is_last})'


class SirenBlock(LinearBlock):
    def __init__(self,
                 in_features,
                 out_features,
                 linear_cls=nn.Linear,
                 w0=30,
                 bias=True,
                 is_first=False,
                 is_last=False):
        super().__init__(
            in_features,
            out_features,
            linear_cls,
            partial(Sine, w0),
            bias,
            is_first,
            is_last
        )
        self.w0 = w0
        self.init_weights()

    def init_weights(self):
        if self.is_first:
            b = 1 / self.in_f
        else:
            b = sqrt(6 / self.in_f) / self.w0

        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-b, b)


class BatchSeparatedLinear(nn.Module):
    def __init__(self, in_feat, out_feat, batch_size=1, bias=True):
        super().__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.batch_size = batch_size

        self.weight = nn.Parameter(torch.Tensor(batch_size, out_feat, in_feat))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(batch_size, out_feat))
        else:
            self.bias = None

        self.init_weights()

    def init_weights(self):
        for i in range(self.batch_size):
            w = self.weight[i]
            nn.init.kaiming_uniform_(w, a=math.sqrt(5))
            if self.bias is not None:
                b = self.bias[i]
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(w)
                bound = 1 / fan_in
                nn.init.uniform_(b, -bound, bound)

    def forward(self, x):
        x = x.transpose(1, -1)
        orig_shape = x.shape
        x = x.reshape(x.size(0), x.size(1), -1)

        out = torch.bmm(self.weight, x)
        if self.bias is not None:
            out += self.bias.unsqueeze(-1)

        out = out.view((out.size(0), self.weight.shape[1]) + orig_shape[2:])
        out = out.transpose(1, -1)

        return out

    def get_layer_by_index(self, idx):
        linear = nn.Linear(self.in_feat, self.out_feat, bias=self.bias is not None)
        linear.weight.data = self.weight[idx].data
        if self.bias is not None:
            linear.bias.data = self.bias[idx].data
        return linear

    def get_layers(self):
        return list(map(self.get_layer_by_index, range(self.batch_size)))


class BaseBlockFactory:
    def __call__(self, in_feat, out_feat, is_first=False, is_last=False):
        raise NotImplementedError


class LinearBlockFactory(BaseBlockFactory):
    def __init__(self, linear_cls=nn.Linear, activation_cls=nn.ReLU, bias=True):
        self.linear_cls = linear_cls
        self.activation_cls = activation_cls
        self.bias = bias

    def __call__(self, in_f, out_f, is_first=False, is_last=False):
        return LinearBlock(in_f, out_f, self.linear_cls, self.activation_cls, self.bias, is_first, is_last)


class SirenBlockFactory(BaseBlockFactory):
    def __init__(self, linear_cls=nn.Linear, w0=30, bias=True):
        self.linear_cls = linear_cls
        self.w0 = w0
        self.bias = bias

    def __call__(self, in_f, out_f, is_first=False, is_last=False):
        return SirenBlock(in_f, out_f, self.linear_cls, self.w0, self.bias, is_first, is_last)


class MLP(nn.Module):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dims: List[int],
                 block_factory: BaseBlockFactory,
                 dropout: float = 0.0,
                 final_activation: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.hidden_dims = hidden_dims or []
        self.block_factory = block_factory
        self.dropout = dropout

        self.blocks = nn.ModuleList()

        in_dims = [self.in_dim] + self.hidden_dims
        out_dims = self.hidden_dims + [self.out_dim]

        for i, (in_feat, out_feat) in enumerate(zip(in_dims, out_dims)):
            is_first = i == 0
            is_last = i == len(self.hidden_dims)

            curr_block = [block_factory(
                in_feat,
                out_feat,
                is_first=is_first,
                is_last=is_last
            )]
            if not is_last and dropout:
                curr_block.append(nn.Dropout(dropout))

            self.blocks.append(nn.Sequential(*curr_block))

        self.final_activation = final_activation
        if final_activation is None:
            self.final_activation = nn.Identity()

    def forward(self, x, modulations=None):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if modulations is not None and len(self.blocks) > i + 1:
                x *= utils.spatial_unsqueeze(modulations[i], num_dims=x.ndim - 2)
        return self.final_activation(x)


class BatchSeparatedMLP(MLP):
    def __init__(self, batch_size: int, block_factory: BaseBlockFactory, *args, **kwargs):

        block_factory = copy(block_factory)
        block_factory.linear_cls = partial(BatchSeparatedLinear, batch_size=batch_size)

        super().__init__(*args, block_factory=block_factory, **kwargs)

        self.block_factory = block_factory
        self.batch_size = batch_size

    def forward(self, x, *args, **kwargs):
        if len(x) != self.batch_size:
            raise ValueError(f'Batch size of input ({len(x)}) must match model\'s batch size ({self.batch_size}) for '
                             f'batch separated model.')

        return super().forward(x, *args, **kwargs)

    @classmethod
    def from_mlp(cls, mlp: MLP, batch_size: int):
        model = cls(
            batch_size,
            block_factory=mlp.block_factory,
            in_dim=mlp.in_dim,
            out_dim=mlp.out_dim,
            hidden_dims=mlp.hidden_dims,
            dropout=mlp.dropout,
            final_activation=mlp.final_activation
        )

        if isinstance(mlp, cls):
            raise AssertionError('Cannot copy from batch separated model. Deep copy the model directly instead.')

        for src, trg in zip(mlp.parameters(), model.parameters()):
            trg.data.copy_(
                utils.batch_expand(src, len(trg))
            )

        return model

    def get_model_by_index(self, idx):
        model = MLP(
            self.in_dim,
            self.out_dim,
            self.hidden_dims,
            self.block_factory,
            self.dropout,
            self.final_activation
        )
        for src_block, trg_block in zip(self.blocks, model.blocks):
            if hasattr(src_block, 'linear'):
                trg_block.linear = src_block.linear.get_layer_by_index(idx)
        return model

    def get_model_splits(self):
        return list(map(self.get_model_by_index, range(self.batch_size)))


class ImplicitNeuralRepresentation(nn.Module):
    def __init__(self, encoder: encodings.CoordinateEncoding, mlp: MLP):
        super().__init__()

        if encoder.out_dim != mlp.in_dim:
            raise AssertionError(f'Number of output dimension of encoder does not match input dimension of MLP '
                                 f'({encoder.out_dim} != {mlp.in_dim})')

        self.encoder = encoder
        self.mlp = mlp
        self.last_input = None
        self.last_encoded_input = None

    def forward(self, x, modulations=None):
        if self.last_input is x and not self.encoder.is_trainable:
            x = self.last_encoded_input
        else:
            self.last_input = x
            x = self.encoder(x)
            self.last_encoded_input = x

        x = self.mlp(x, modulations)
        return x


class ModulationNetwork(nn.Module):
    def __init__(self, in_dim: int, mod_dims: List[int], activation=nn.ReLU):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(len(mod_dims)):
            self.blocks.append(nn.Sequential(
                nn.Linear(in_dim + (mod_dims[i - 1] if i else 0), mod_dims[i]),
                activation()
            ))

    def forward(self, input):
        out = input
        mods = []
        for block in self.blocks:
            out = block(out)
            mods.append(out)
            out = torch.cat([out, input], dim=-1)
        return mods


class ImplicitDecoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 out_dim: int,
                 hidden_dims: List[int],
                 block_factory: BaseBlockFactory,
                 input_encoder: encodings.CoordinateEncoding = None,
                 modulation: bool = False,
                 dropout: float = 0.0,
                 final_activation=torch.sigmoid):
        super().__init__()

        self.input_encoder = input_encoder
        self.latent_dim = latent_dim

        self.mod_network = None
        if modulation:
            self.mod_network = ModulationNetwork(
                in_dim=latent_dim,
                mod_dims=hidden_dims,
                activation=nn.ReLU
            )

        self.net = MLP(
            in_dim=input_encoder.out_dim + latent_dim * (not modulation),
            out_dim=out_dim,
            hidden_dims=hidden_dims,
            block_factory=block_factory,
            dropout=dropout,
            final_activation=final_activation
        )

    def forward(self, input, latent):
        if self.input_encoder is not None:
            input = self.input_encoder(input)

        if self.mod_network is None:
            b, *spatial_dims, c = input.shape
            latent = latent.view(b, *((1,) * len(spatial_dims)), -1).repeat(1, *spatial_dims, 1)
            out = self.net(torch.cat([latent, input], dim=-1))
        else:
            mods = self.mod_network(latent)
            out = self.net(input, mods)

        return out
