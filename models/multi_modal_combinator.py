import einops as einops
import numpy as np
import torch
import torch.nn.functional

import models.utils


class SimpleCombinator(torch.nn.Module):

    def __init__(self, mode_in_features, mode_count, weight_sharing=False, bin_count=None, interval_type=None):
        super(SimpleCombinator, self).__init__()
        self._weight_sharing = weight_sharing
        if not self._weight_sharing:
            mode_in_features = mode_in_features * mode_count
        self._bin_count = bin_count
        self._interval_type = interval_type
        self._backbone = interval_network(interval_type=interval_type, bin_count=bin_count,
                                          in_features=mode_in_features)
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, x, temp=None):
        assert not self._weight_sharing, 'weight sharing is not supported for simple multi-modal combinator'
        batch_size, *_ = x.size()
        x = x.view(batch_size, -1)
        # x = self._backbone(x)
        # x = x.view(batch_size, -1)
        # x = x.sum(dim=-1)
        x_interval = self._backbone(x)
        x = interval_calculation(x_interval, self._bin_count, self._interval_type)
        if temp:
            x = x / temp
        if self._interval_type == 'hierarchical':
            return (self._sigmoid(x), x_interval[..., :self._bin_count])
        if self._interval_type == 'double hierarchical':
            return (self._sigmoid(x), x_interval)
        return self._sigmoid(x)


class ModalityAttentionCombinator(torch.nn.Module):

    def __init__(self, mode_in_features, weight_sharing=False, bin_count=None, interval_type=None):
        super(ModalityAttentionCombinator, self).__init__()
        self._in_features = mode_in_features
        self._common_encoder = torch.nn.Identity()
        if weight_sharing:
            self._common_encoder = torch.nn.Sequential(
                torch.nn.Linear(mode_in_features, int(mode_in_features // 2), bias=False),
                torch.nn.ReLU()
            )
            mode_in_features = int(mode_in_features // 2)
        self._attention = torch.nn.Linear(mode_in_features, 1)
        self._interval_type = interval_type
        self._bin_count = bin_count
        self._backbone = interval_network(interval_type=self._interval_type, bin_count=self._bin_count,
                                          in_features=mode_in_features)
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, x, temp=None):
        batch_size, *_ = x.size()
        x = self._common_encoder(x)
        attention_weights = self._attention(x).squeeze(dim=-1)
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
        x = (attention_weights[..., None] * x).sum(dim=-2)
        x_interval = self._backbone(x)
        x = interval_calculation(x_interval, self._bin_count, self._interval_type)
        if temp:
            x = x / temp
        if self._interval_type == 'hierarchical':
            return (self._sigmoid(x), x_interval[..., :self._bin_count])
        if self._interval_type == 'double hierarchical':
            return (self._sigmoid(x), x_interval)
        return self._sigmoid(x)


class TransformerCombinator(torch.nn.Module):

    def __init__(self, mode_in_features, weight_sharing=False, cls_token=False, pool='mean', n_head=1, pos=False,
                 bin_count=None, interval_type=None):
        super(TransformerCombinator, self).__init__()
        self._pool = pool
        self._in_features = mode_in_features
        self._weight_sharing = weight_sharing
        self._cls_token = torch.nn.Parameter(torch.randn(1, 1, self._in_features)) if cls_token else None
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self._in_features,
                                                         nhead=n_head,
                                                         dim_feedforward=self._in_features * 2)
        self._transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        self._pos = models.utils.PositionalEncoding(self._in_features, max_len=3) if pos else None
        if self._weight_sharing:
            self._transformer = torch.nn.Linear(self._in_features, self._in_features)
        self._interval_type = interval_type
        self._bin_count = bin_count
        self._backbone = interval_network(interval_type=self._interval_type, bin_count=self._bin_count,
                                          in_features=self._in_features)
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, x, temp=None):
        batch_size, *_ = x.size()
        if not self._weight_sharing:
            if self._cls_token is not None:
                cls_token = einops.repeat(self._cls_token, '1 1 d -> b 1 d', b=batch_size)
                x = torch.cat((cls_token, x), dim=1)
            if self._pos is not None:
                x = self._pos(x)
            x = self._transformer(x)
        else:
            x = self._transformer(x)
            attention = torch.bmm(x, einops.rearrange(x, 'b s d -> b d s'))
            attention = torch.nn.functional.softmax(attention / np.sqrt(self._in_features), dim=-1)
            x = torch.bmm(attention, x)
        x = x.mean(dim=1) if self._pool == 'mean' else x[:, 0]
        x_interval = self._backbone(x)
        x = interval_calculation(x_interval, self._bin_count, self._interval_type)
        if temp:
            x = x / temp
        if self._interval_type == 'hierarchical':
            return (self._sigmoid(x), x_interval[..., :self._bin_count])
        if self._interval_type == 'double hierarchical':
            return (self._sigmoid(x), x_interval)
        return self._sigmoid(x)


def interval_network(interval_type, bin_count, in_features):
    if interval_type is None:
        return torch.nn.Linear(in_features, 1)
    if interval_type == 'chen':
        return torch.nn.Linear(in_features, bin_count)
    if interval_type == 'confidence':
        return torch.nn.Linear(in_features, 2)
    if interval_type == 'hierarchical':
        return torch.nn.Linear(in_features, bin_count + 1)
    if interval_type == 'double hierarchical':
        return torch.nn.Linear(in_features, bin_count)


def interval_calculation(pred, bin_count, interval_type):
    if interval_type is None:
        return pred
    if interval_type == 'chen':
        return pred
    if interval_type == 'confidence':
        return torch.normal(mean=pred[..., 0], std=torch.sigmoid(pred[..., 1]))
    if interval_type == 'hierarchical':
        length = 1 / bin_count
        choice = torch.topk(pred[..., :bin_count], 1, dim=-1)[1]
        choice = bin_count - choice - 1
        output = choice.squeeze() * length + (1 + torch.tanh(pred[..., -1])) * length / 2
        return output
    if interval_type == 'double hierarchical':
        return torch.topk(pred[..., :bin_count], 1, dim=-1)[0]
    raise RuntimeError(f'invalid interval type {interval_type}')
