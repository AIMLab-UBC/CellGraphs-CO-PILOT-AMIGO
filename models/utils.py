import math

import torch


# Credit:
# https://github.com/dmlc/dgl/blob/148575e4895eecb28880d818d3e99d248dd3d070/examples/pytorch/sagpool/utils.py#L49


def get_batch_id(num_nodes: torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.
    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full((num_nodes[i],), i, dtype=torch.long, device=num_nodes.device)
        batch_ids.append(item)
    return torch.cat(batch_ids)


# Credit:
# https://github.com/dmlc/dgl/blob/148575e4895eecb28880d818d3e99d248dd3d070/examples/pytorch/sagpool/utils.py#L65
def topk(x: torch.Tensor, ratio: float, batch_id: torch.Tensor, num_nodes: torch.Tensor):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.
    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.

    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat([num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0)

    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full((batch_size * max_num_nodes,), torch.finfo(x.dtype).min)
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [torch.arange(k[i], dtype=torch.long, device=x.device) + i * max_num_nodes for i in range(batch_size)]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k


# Credit:
# https://github.com/mahmoodlab/PathomicFusion/blob/3d8b9f3f6221d27ade71c5d04b1af92e28e0b11e/utils.py#L225
def init_max_weights(module):
    for m in module.modules():
        if type(m) == torch.nn.Linear:
            stdv = 1. / math.sqrt(m.weight.size(1))
            m.weight.data.normal_(0, stdv)
            if m.bias is not None:
                m.bias.data.zero_()


def sum_with_index(feature, index, batch_size):
    dtype = feature.dtype
    device = feature.device

    feature = feature.type(torch.double)
    index = index.type(torch.LongTensor).to(device)
    if len(feature.size()) == 1:
        feature = feature.reshape(-1, 1)

    correspondence = torch.zeros((batch_size, feature.size(0)))
    for i in range(correspondence.size(0)):
        correspondence[i] = (index == i).squeeze()
    correspondence = correspondence.type(torch.double).to(device)
    output = torch.matmul(correspondence, feature).type(dtype)
    return output

    # non-deterministic
    # output = torch.zeros((batch_size, feature.size(1)), dtype=feature.dtype).to(feature.device)
    # output.index_add_(0, index, feature)
    # return output


def aggregate_with_index(feature, index, batch_size):
    index = index.type(torch.LongTensor).to(feature.device)
    output = sum_with_index(feature, index, batch_size)
    output = output / torch.bincount(index).reshape(-1, 1)
    return output.squeeze(dim=-1)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(0), :]
        return self.dropout(x)

