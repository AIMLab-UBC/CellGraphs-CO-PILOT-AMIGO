from abc import abstractmethod
import dgl.nn.pytorch
import torch
import torch.nn.functional

import models.utils
import utils.utils


class IdentityPool(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def forward(self, graph, x):
        return graph, x, torch.arange(graph.num_nodes())

# Credit:
# https://github.com/dmlc/dgl/blob/148575e4895eecb28880d818d3e99d248dd3d070/examples/pytorch/sagpool/layer.py#L8
class SAGPool(torch.nn.Module):
    """The Self-Attention Pooling layer in paper
    `Self Attention Graph Pooling <https://arxiv.org/pdf/1904.08082.pdf>`
    Args:
        in_dim (int): The dimension of node feature.
        ratio (float, optional): The pool ratio which determines the amount of nodes
            remain after pooling. (default: :obj:`0.5`)
        conv_op (torch.nn.Module, optional): The graph convolution layer in dgl used to
        compute scale for each node. (default: :obj:`dgl.nn.GraphConv`)
        non_linearity (Callable, optional): The non-linearity function, a pytorch function.
            (default: :obj:`torch.tanh`)
    """

    def __init__(self, in_dim: int, ratio=0.5, conv_op=dgl.nn.GraphConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.score_layer = conv_op(in_dim, 1, norm='none')
        self.non_linearity = non_linearity

    def forward(self, graph: dgl.DGLGraph, feature: torch.Tensor):
        score = self.score_layer(graph, feature).squeeze()
        perm, next_batch_num_nodes = models.utils.topk(
            score,
            self.ratio,
            models.utils.get_batch_id(graph.batch_num_nodes()),
            graph.batch_num_nodes()
        )
        feature = self.feature_transform(graph, feature, score, perm)

        batch_edges_bin = torch.nn.functional.pad(torch.cumsum(graph.batch_num_edges(), dim=0), (1, 0), "constant", 0)
        graph = dgl.node_subgraph(graph, perm)
        next_batch_num_edges = torch.histogram(graph.edata[dgl.EID].cpu().type(torch.FloatTensor), batch_edges_bin.cpu().type(torch.FloatTensor))[0]
        next_batch_num_edges = next_batch_num_edges.type(torch.LongTensor).to(graph.device)

        # node_subgraph currently does not support batch-graph,
        # the 'batch_num_nodes' of the result subgraph is None.
        # So we manually set the 'batch_num_nodes' here.
        # Since global pooling has nothing to do with 'batch_num_edges',
        # we can leave it to be None or unchanged.
        graph.set_batch_num_nodes(next_batch_num_nodes)
        graph.set_batch_num_edges(next_batch_num_edges)

        assert batch_edges_bin.size(0) - 1 == graph.batch_num_edges().size(0)
        utils.utils.check_batch_validity(graph)
        return graph, feature, perm

    def feature_transform(self, graph, feature, score, perm):
        feature = feature[perm] * self.non_linearity(score[perm]).view(-1, 1)
        return feature


class TransformerSAGPool(SAGPool):
    def __init__(self, in_dim: int, *args, **kwargs):
        super(TransformerSAGPool, self).__init__(*args, in_dim=in_dim, **kwargs)
        self.transformer = dgl.nn.pytorch.glob.SetTransformerEncoder(d_model=in_dim, n_heads=2, d_head=in_dim//2, d_ff=in_dim//2, n_layers=1, block_type='isab', m=in_dim//2)

    def forward(self, graph: dgl.DGLGraph, feature: torch.Tensor):
        feature = self.transformer(graph, feature)
        return super().forward(graph, feature)


class SimilarityScorer(torch.nn.Module):

    def forward(self, graph, features):
        def message_fn(edges):
            return {'m': torch.nn.CosineSimilarity(dim=-1)(edges.src['h'], edges.dst['h'])}
        with graph.local_scope():
            graph = dgl.remove_self_loop(graph)
            graph.ndata['h'] = features
            graph.update_all(message_fn, dgl.function.mean('m', 'h'))
            return graph.ndata['h']



class LanderSAGPool(SAGPool):

    def __init__(self, in_dim, non_linearity=torch.nn.Identity(), *args, **kwargs):
        super().__init__(*args, in_dim=in_dim, non_linearity=non_linearity, **kwargs)
        # self.score_layer = self.DensityScorer()
        density_scorer = SimilarityScorer()
        self.score_layer = self.ScoreCombiner(gnn_scorer=self.score_layer, density_scorer=density_scorer)

    class ScoreCombiner(torch.nn.Module):

        def __init__(self, gnn_scorer, density_scorer) -> None:
            super().__init__()
            self.gnn_scorer = gnn_scorer
            self.density_scorer = density_scorer
            # self.post_process = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1), torch.nn.Sigmoid())
        
        def forward(self, graph, features):
            # return self.post_process((torch.sigmoid(self.gnn_scorer(graph, features)).squeeze(dim=-1), self.density_scorer), dim=1)
            return torch.sigmoid(self.gnn_scorer(graph, features)).squeeze(dim=-1) / (self.density_scorer(graph, features) + 1e-8)


class LanderTransformerSAGPool(LanderSAGPool,TransformerSAGPool):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class RandomPool(SAGPool):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_layer = self.Random_scorer()
    
    class Random_scorer(torch.nn.Module):
        def forward(self, graph, feature):
            return torch.rand(feature.size(0)).to(feature.device)
