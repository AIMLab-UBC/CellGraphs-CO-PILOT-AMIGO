import copy
import dgl
import dgl.function
import dgl.nn.pytorch.conv
import dgl.utils
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.factory import SegmentedKNNGraph
import numpy as np
import wandb
from graph_visualization import draw_graph

import torch
import torch.nn
import torch.nn.functional
from histocartography.preprocessing.graph_builders import FEATURES

import models.pooling
import models.density
from models.positional_encoding import NonePositinEncoding, RandlaGraphConv, get_position_encoding
import models.utils
import models.norm
import utils.utils
import utils.graph
from data_workflow.histocartography_extension.graph_builder import EDGE_WEIGHT, FEATURES, NORM_CENTROID, CENTROID
from data_workflow.histocartography_extension.graph_builder import TYPE as TYPE_FIELD


def get_conv_type(conv_type):
    if conv_type == 'sage':
        return dgl.nn.pytorch.conv.SAGEConv, {'aggregator_type': 'mean'}
    if conv_type == 'no_self_sage':
        return NoSelfSAGEConv, {'aggregator_type': 'mean'}
    if conv_type == 'gat':
        return dgl.nn.pytorch.conv.GATConv, {'residual': True, 'num_heads': 1}
    if conv_type == 'shared_sage':
        return SharedSAGEConv, {'aggregator_type': 'mean'}
    if conv_type == 'edge_conv':
        return dgl.nn.pytorch.conv.EdgeConv, {'batch_norm': False}
    if conv_type == 'concat_edge_conv':
        return ConcatEdgeConv, {'batch_norm': False}
    if conv_type == 'residual_sage':
        return ResidualSAGEConv, {'aggregator_type': 'mean'}
    if conv_type == 'conv':
        return dgl.nn.pytorch.conv.GraphConv, {'norm': 'none', 'weight': True, 'bias': True}
    if conv_type == 'gat':
        return dgl.nn.pytorch.conv.GATConv, {'residual': True, 'num_heads': 1}
    if conv_type == 'randla_conv':
        return RandlaGraphConv, {}
    raise RuntimeError('invalid conv type')


def build_knn_graph(graph, dynamic_graph, fully_connected=False):
    if not dynamic_graph:
        return graph

    centers = graph.ndata[NORM_CENTROID]
    features = graph.ndata[FEATURES]

    # NOTE: should consider batch, be bidirectional, have self-loop, and keep order of the features and nodes the same
    with torch.no_grad():

        if fully_connected:
            graph = utils.graph.batch_fully_connected_graph(graph.batch_num_nodes())
        else:
            device = graph.device
            num_nodes = graph.batch_num_nodes()

            node_offsets = torch.nn.functional.pad(torch.cumsum(num_nodes, dim=0), (1, 0), 'constant', 0)
            missing_count = torch.clip(dynamic_graph-num_nodes, min=0)
            new_centers = torch.cat([torch.cat((centers[node_offsets[i]:node_offsets[i]+num_nodes[i]], torch.zeros((missing_count[i], centers.size(1)), device=device)), dim=0) for i in range(num_nodes.size(0))], dim=0)
            
            # TODO: double check this
            offset = torch.nn.functional.pad(torch.cumsum(missing_count, dim=0), (1, 0), 'constant', 0)
            invalid_node_ids = torch.cat([torch.arange(missing_count[i], device=device) + node_offsets[i+1] + offset[i] for i in range(num_nodes.size(0))])

            graph = dgl.segmented_knn_graph(x=new_centers, k=dynamic_graph, segs=(num_nodes + missing_count).tolist(), algorithm='bruteforce-blas')
            graph = dgl.remove_nodes(graph, invalid_node_ids)
        
            graph.set_batch_num_nodes(num_nodes)
            graph.set_batch_num_edges(num_nodes * torch.clip(num_nodes, max=dynamic_graph))  # some nodes might have less than 

            # make the graph bidirectional
            graph = utils.utils.batch_to_bidirected(graph)

    graph.ndata[FEATURES] = features
    graph.ndata[NORM_CENTROID] = centers

    utils.utils.check_batch_validity(graph)
    return graph


class PathomicGraphNet(torch.nn.Module):
    def __init__(self,
                 features=1036,
                 nhid=256,
                 grph_dim=128,
                 dropout_rate=0.25,
                 use_edges=0,
                 pooling_ratio=0.20,
                 act=torch.nn.Sigmoid(),
                 label_dim=1,
                 init_max=True,
                 edge_weighting=None,
                 conv_type='sage',
                 learnable_skip_connection=False,
                 ntypes=None,
                 etypes=None,
                 weight_sharing=False,
                 node_type_count=None,
                 shared_module_weights=None,
                 position_encoding=None,
                 preconv=False,
                 dynamic_graph=None,
                 dynamic_graph_expansion_scale=0,
                 expansion_pos_encoding=False,
                 pair_norm=False,
                 pooling_type='sag',
                 layer_sharing=False,
                 gated_attention='',
                 intra_sharing=False,
                 extra_layer=False,
                 expansion_dim_factor=1,
                 plot_graph=False,
                 similarity_encoding=False,
                 hierarchical_attention=False,
                 n_layers=3,
                 fully_connected=False,
                 multi_block=1,
                 single_layer_preconv=False,
                 ):
        super(PathomicGraphNet, self).__init__()

        self.dropout_rate = dropout_rate
        self.use_edges = use_edges
        self.act = act
        self.edge_weighting = edge_weighting
        self.dynamic_graph = dynamic_graph
        self.dynamic_graph_expansion_scale = dynamic_graph_expansion_scale
        self.pair_norm = pair_norm
        self.gated_attention = gated_attention
        self.extra_layer = extra_layer
        self.plot_graph = plot_graph
        self.hierarchical_attention = hierarchical_attention
        self.fully_connected = fully_connected
        

        assert not self.hierarchical_attention or self.gated_attention, 'turn off hierarchical attention or use gated attention too'

        # l1_input_size = features if not layer_sharing and not intra_sharing else nhid
        l1_input_size = features if not preconv else nhid
        hidden_feature = features // 2
        if single_layer_preconv:
            self.preconv = torch.nn.Sequential(torch.nn.Linear(features, l1_input_size), torch.nn.BatchNorm1d(l1_input_size), torch.nn.GELU(),) if preconv else torch.nn.Identity()
        else:
            self.preconv = torch.nn.Sequential(torch.nn.Linear(features, hidden_feature), torch.nn.BatchNorm1d(hidden_feature), torch.nn.GELU(), torch.nn.Linear(hidden_feature, l1_input_size), torch.nn.BatchNorm1d(l1_input_size), torch.nn.GELU(),) if preconv else torch.nn.Identity()
        # self.preconv = torch.nn.Sequential(torch.nn.Linear(features, l1_input_size), torch.nn.BatchNorm1d(l1_input_size), torch.nn.GELU()) if preconv else torch.nn.Identity()
        
        # Get position encodings
        self.create_position_encodings(nhid, position_encoding, expansion_pos_encoding, layer_sharing, intra_sharing, expansion_dim_factor, n_layers, l1_input_size)

        # Get similarity encodings
        self.create_similarity_encodings(nhid, similarity_encoding, n_layers)

        assert not intra_sharing or conv_type == 'shared_sage', 'you should use shared_sage in intra_sharing mode'
        assert not intra_sharing or not weight_sharing, 'one of weight_sharing or intra_sharing should be enabled'

        if intra_sharing:
            assert self.conv_input_sizes.count(self.conv_input_sizes[0]) == n_layers, 'input features to the layers should be equal'

            neigh_layer = torch.nn.Linear(self.conv_input_sizes[0], nhid//4, bias=False)
            self_layer = torch.nn.Linear(self.conv_input_sizes[0], nhid//4, bias=False)

            shared_module_weights = [{'neigh_shared_weight': neigh_layer, 'self_shared_weight': self_layer}] * n_layers

        # Get convolution layers
        self.create_convs(nhid, conv_type, weight_sharing, shared_module_weights, intra_sharing, n_layers)

        # Get pooling layers
        self.create_pools(nhid, pooling_ratio, pooling_type, n_layers)

        if self.gated_attention:
            self.create_gated_attentions(nhid, n_layers)

        if self.edge_weighting == 'similarity_attention' or self.edge_weighting == 'similarity_dual_attention' or self.edge_weighting == 'dual_attention':
            self.similarity_mlps = torch.nn.ModuleList([torch.nn.Linear(self.conv_input_sizes[i], nhid, bias=False) for i in range(n_layers)])
            if self.edge_weighting == 'similarity_dual_attention' or self.edge_weighting == 'dual_attention':
                self.similarity_dual_mlps = torch.nn.ModuleList([torch.nn.Linear(self.conv_input_sizes[i], nhid, bias=False) for i in range(n_layers)])


        if layer_sharing:
            for i in range(1, n_layers):
                self.convs[i] = self.convs[0]
                self.pools[i] = self.pools[0]
                self.position_encodings[i] = self.position_encodings[0]
                self.similarity_encodings[i] = self.similarity_encodings[0]
                if self.gated_attention:
                    self.gated_attentions[i] = self.gated_attentions[0]
                if self.edge_weighting == 'similarity_attention':
                    self.similarity_mlps[i] = self.similarity_mlps[0]

        # temp changes
        ################################################################
        print('multi_block', multi_block)
        if multi_block > 1:
            # 3 by 2 block sharing
            for i in range(1, n_layers):
                target_idx = int((i // multi_block) * multi_block)
                print('id', i, 'target', target_idx)
                self.convs[i] = self.convs[target_idx]
                self.pools[i] = self.pools[target_idx]
                self.position_encodings[i] = self.position_encodings[target_idx]
                self.similarity_encodings[i] = self.similarity_encodings[target_idx]
                if self.gated_attention:
                    self.gated_attentions[i] = self.gated_attentions[target_idx]
                if self.edge_weighting == 'similarity_attention' or self.edge_weighting == 'similarity_dual_attention' or self.edge_weighting == 'dual_attention':
                    self.similarity_mlps[i] = self.similarity_mlps[target_idx]
                    if self.edge_weighting == 'similarity_dual_attention' or self.edge_weighting == 'dual_attention':
                        self.similarity_dual_mlps[i] = self.similarity_dual_mlps[target_idx]

        self.skip_weight = None
        if learnable_skip_connection:
            self.skip_weight = torch.nn.Linear(n_layers * nhid * (2 if not self.gated_attetnion else 1), 2 * nhid)

        self.lin1 = torch.nn.Linear(nhid * (2 if not self.gated_attention else 1), nhid)
        self.lin2 = torch.nn.Linear(nhid, grph_dim)
        self.lin3 = torch.nn.Linear(grph_dim, label_dim)

        # self.output_range = torch.nn.Parameter(torch.FloatTensor([6]), requires_grad=False)
        # self.output_shift = torch.nn.Parameter(torch.FloatTensor([-3]), requires_grad=False)

        self.max_pool = dgl.nn.pytorch.glob.MaxPooling()
        self.avg_pool = dgl.nn.pytorch.glob.AvgPooling()

        self.temp_graph = []
        self.edge_w = torch.nn.Parameter(torch.FloatTensor([1]))
        if init_max:
            models.utils.init_max_weights(self)

    def forward(self, graph: dgl.DGLGraph):
        graph_viz = []
        with graph.local_scope():
            graph = utils.utils.add_batch_self_loop(graph)

            feature_tracker = []

            graph.ndata[FEATURES] = _feature_normalize(graph.ndata[FEATURES])
            x = self.preconv(graph.ndata[FEATURES])

            for i in range(len(self.convs)):

                # Build Knn Graph
                graph = build_knn_graph(graph, self._expand_dynamic_graph(i), self.fully_connected)

                # Encode positions
                x = self.position_encodings[i](graph, x)

                # Encode similarity
                x = self.similarity_encodings[i](graph, x)

                # Get edge weights
                edge_weight = self._get_edge_weight(graph, x, i)
                if edge_weight is not None:
                    edge_weight = dgl.nn.functional.edge_softmax(graph, edge_weight)

                # Run convolution
                if isinstance(self.convs[i], dgl.nn.pytorch.conv.GATConv):
                    x = self.convs[i](graph, x).squeeze(-2)
                else:
                    x = self.convs[i](graph, x, edge_weight=edge_weight)
                x = torch.nn.functional.gelu(self._pair_normalize(graph, x))

                # Perform pooling
                graph, x, _ = self.pools[i](graph, x)

                # Plot the graph
                if self.plot_graph:
                    graph_viz.append(self.plot(graph, idx=0))
                
                # Graph pooling
                feature_tracker.append((graph, x)) # might need to copy the graph to decouple references

            residuals = []
            if not self.hierarchical_attention:
                for i, (graph, x) in enumerate(feature_tracker):
                    residuals.append(torch.cat([self.max_pool(graph, x), self.avg_pool(graph, x)], dim=1) if not self.gated_attention else self.gated_attentions[i](graph, x))
            else:
                condition = None
                for i in range(len(self.convs)-1, -1, -1):
                    graph, x = feature_tracker[i]
                    condition = self.gated_attentions[i](graph, x, condition=condition)
                    residuals.append(condition)
                residuals = residuals[::-1]
            print('residuals distribution', [d.mean() for d in residuals])

            if self.skip_weight is None:
                # x = residuals[2] + residuals[-1]
                x = sum(residuals)
            else:
                x = torch.cat(residuals, dim=1).to(x.device)  # n, 3, f
                x = self.skip_weight(x)
                # attn = self.skip_weight(x).squeeze(dim=-1)  # n, 3
                # attn = torch.nn.functional.softmax(attn, dim=-1)  # n, 3
                # x = (attn[..., None] * x).sum(dim=1)

            x = torch.nn.functional.gelu(self.lin1(x))
            x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
            features = torch.nn.functional.gelu(self.lin2(x))
            out = self.lin3(features)
            if self.act is not None:
                out = self.act(out)

                # if isinstance(self.act, torch.nn.Sigmoid):
                #     out = out * self.output_range + self.output_shift

            return features, out, graph_viz

    def create_pools(self, nhid, pooling_ratio, pooling_type, n_layers):

        def make_pooling(pooling):
            if pooling is None:
                return models.pooling.IdentityPool
            if pooling == 'transformersag':
                return models.pooling.TransformerSAGPool
            elif pooling == 'sag':
                return models.pooling.SAGPool
            elif pooling == 'densitytransformersag':
                return models.density.DensityTransformerSAGPool
            elif pooling == 'landersag':
                return models.pooling.LanderSAGPool
            elif pooling == 'densitysag':
                return models.density.DensitySAGPool
            elif pooling == 'landertransformersag':
                return models.pooling.LanderTransformerSAGPool
            elif pooling == 'random':
                return models.pooling.RandomPool
            raise ValueError(f'unknown pooling type {pooling}')
    
        self.pools = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.pools.append(make_pooling(pooling_type)(in_dim=nhid, ratio=pooling_ratio))

    def create_gated_attentions(self, nhid, n_layers):

        def make_attention(nhid):

            if self.gated_attention == 'simple_mean':
                return SimpleGatedAttention(in_feat=nhid, nhid=nhid, condition_enable=self.hierarchical_attention)
            elif self.gated_attention == 'mean':
                return GatedAttention(in_feat=nhid, nhid=nhid, condition_enable=self.hierarchical_attention)
            elif self.gated_attention.startswith('var'):
                n_head = int(self.gated_attention[3:]) if len(self.gated_attention) > 3 else 1
                return GatedVarAttention(in_feat=nhid, nhid=nhid, n_head=n_head, condition_enable=self.hierarchical_attention)
            raise RuntimeError('invalid gated pooling type')
    
        self.gated_attentions = torch.nn.ModuleList()
        for _ in range(n_layers):
            self.gated_attentions.append(make_attention(nhid))    
    
    def create_convs(self, nhid, conv_type, weight_sharing, shared_module_weights, intra_sharing, n_layers):
        self.convs = torch.nn.ModuleList()
        if conv_type != 'gin':
            conv_type, kwargs = get_conv_type(conv_type)
            for i in range(n_layers):
                new_args = {k: v for k, v in kwargs.items()}
                new_args.update(shared_module_weights[i] if weight_sharing or intra_sharing and len(shared_module_weights) > i else {})
                conv = conv_type(self.conv_input_sizes[i], nhid, **new_args)
                self.convs.append(conv)
        else:
            for i in range(n_layers):
                conv = dgl.nn.pytorch.GINConv(torch.nn.Sequential(torch.nn.Linear(self.conv_input_sizes[i], nhid), torch.nn.BatchNorm1d(nhid), torch.nn.ReLU(), torch.nn.Linear(nhid, nhid)), aggregator_type='sum')
                self.convs.append(conv)

    def create_similarity_encodings(self, nhid, similarity_encoding_type, n_layers):
        self.similarity_encodings = torch.nn.ModuleList()
        for i in range(n_layers):
            similarity_encoding, input_size = get_similarity_encoding(similarity_encoding_type, in_feat=self.conv_input_sizes[i], out_feat=self.conv_input_sizes[i]+nhid)
            self.similarity_encodings.append(similarity_encoding)
            self.conv_input_sizes[i] = input_size

    def create_position_encodings(self, nhid, position_encoding, expansion_pos_encoding, layer_sharing, intra_sharing, expansion_dim_factor, n_layers, l1_input_size):
        self.conv_input_sizes = []
        self.position_encodings = torch.nn.ModuleList()
        position_input_sizes = [l1_input_size] + [nhid] * (n_layers - 1)
        for i in range(n_layers):
            encoding_type = position_encoding if i == 0 or expansion_pos_encoding else None
            position_encoding, input_size = get_position_encoding(encoding_type, in_feat=position_input_sizes[i], out_feat=int(expansion_dim_factor*nhid))
            self.position_encodings.append(position_encoding)
            self.conv_input_sizes.append(input_size)

    def _get_edge_weight(self, graph, features, layer_id):
        if not self.edge_weighting:
            return None
        if self.edge_weighting == 'similarity':
            with graph.local_scope():
                graph.ndata['h'] = features
                graph.apply_edges(lambda edges: {'weight': torch.nn.CosineSimilarity(dim=-1)(edges.src['h'], edges.dst['h'])})
                return graph.edata['weight']
        elif self.edge_weighting == 'similarity_attention':
            with graph.local_scope():
                graph.ndata['h'] = self.similarity_mlps[layer_id](features)
                graph.apply_edges(lambda edges: {'weight': torch.nn.CosineSimilarity(dim=-1)(edges.src['h'], edges.dst['h'])})
                return graph.edata['weight']
        elif self.edge_weighting == 'similarity_dual_attention':
            with graph.local_scope():
                graph.ndata['h1'] = self.similarity_mlps[layer_id](features)
                graph.ndata['h2'] = self.similarity_dual_mlps[layer_id](features)
                graph.apply_edges(lambda edges: {'weight': torch.nn.CosineSimilarity(dim=-1)(edges.src['h1'], edges.dst['h2'])})
                return graph.edata['weight']
        elif self.edge_weighting == 'dual_attention':
            with graph.local_scope():
                graph.ndata['h1'] = self.similarity_mlps[layer_id](features)
                graph.ndata['h2'] = self.similarity_dual_mlps[layer_id](features)
                graph.apply_edges(lambda edges: {'weight': torch.sum(edges.src['h1'] * edges.dst['h2'], dim=-1)/128})
                return graph.edata['weight']
        elif self.edge_weighting == 'distance':
            with graph.local_scope():
                graph.ndata['h'] = features
                graph.apply_edges(lambda edges: {'weight': torch.norm(edges.src[NORM_CENTROID] - edges.dst[NORM_CENTROID], dim=-1)})
                return torch.exp(-graph.edata['weight'])
        raise ValueError(f'unknown edge weighting type {self.edge_weighting}')
        edge_weight = graph.edata[EDGE_WEIGHT].type(
            torch.cuda.FloatTensor
        )
        edge_weight = (edge_weight - edge_weight.min()) / (edge_weight.max() - edge_weight.min())
        edge_weight = 1 + self.edge_w * torch.exp(-edge_weight)
        return edge_weight

    def _pair_normalize(self, graph, x):
        return models.norm.PairNorm()(graph, x) if self.pair_norm else x

    def _expand_dynamic_graph(self, level):
        factor = self.dynamic_graph_expansion_scale ** level
        return None if not self.dynamic_graph else int(self.dynamic_graph * factor)

    def plot(self, graph, idx):
        from histocartography.visualization import OverlayGraphVisualization, InstanceImageVisualization
        with graph.local_scope():
            with torch.no_grad():
                graph = dgl.unbatch(graph)[idx]
                graph.ndata[CENTROID] = graph.ndata[NORM_CENTROID] * 300
                visualizer = OverlayGraphVisualization(instance_visualizer=InstanceImageVisualization())
                fig = visualizer.process(np.ones((300, 300, 3)), graph.cpu())
                # fig = draw_graph(center_field=NORM_CENTROID, image_file=None, graph_file=graph.cpu(), seg_file=None, sparsity=None, output_path=None)
                return wandb.Image(fig)


class PathomicGraphNetHetero(torch.nn.Module):
    def __init__(self,
                 features=1036,
                 nhid=128,
                 grph_dim=32,
                 dropout_rate=0.25,
                 use_edges=0,
                 pooling_ratio=0.20,
                 act=torch.nn.Sigmoid(),
                 label_dim=1,
                 init_max=True,
                 node_type_count=None,
                 weight_sharing=False,
                 edge_weighting=False,
                 learnable_skip_connection=False,
                 conv_type='sage',
                 etypes=None,
                 shared_module_weights=None,
                 position_encoding=None,
                 dynamic_graph=None,
                 dynamic_graph_expansion_scale=0,
                 expansion_pos_encoding=False,
                 ):
        super(PathomicGraphNetHetero, self).__init__()
        self._weight_sharing = weight_sharing
        self.dropout_rate = dropout_rate
        self.use_edges = use_edges
        self.act = act
        self.position_encoding = position_encoding
        self.dynamic_graph = self.dynamic_graph

        conv_type, kwargs = get_conv_type(conv_type)

        # self.conv1 = create_hetero_layer(SharedSAGEConv, in_feats=features, out_feats=nhid,
        #                                  aggregator_type='mean', etypes=etypes,
        #                                  **self._creat_sage_shared_weights(features, nhid))
        self.conv1 = HeteroSAGEConv(in_feats=features, out_feats=nhid, aggregator_type='mean',
                                    node_type_count=node_type_count, weight_sharing=weight_sharing,
                                    **shared_module_weights[0])
        # self.conv1 = conv_type(features, nhid, **kwargs)
        self.pool1 = models.pooling.SAGPool(nhid, ratio=pooling_ratio)
        # self.conv2 = HeteroSAGEConv(in_feats=nhid, out_feats=nhid, aggregator_type='mean',
        #                            node_type_count=node_type_count, weight_sharing=weight_sharing)
        self.conv2 = conv_type(nhid, nhid, **kwargs)
        self.pool2 = models.pooling.SAGPool(nhid, ratio=pooling_ratio)
        # self.conv3 = HeteroSAGEConv(in_feats=nhid, out_feats=nhid, aggregator_type='mean',
        #                            node_type_count=node_type_count, weight_sharing=weight_sharing)
        self.conv3 = conv_type(nhid, nhid, **kwargs)
        self.pool3 = models.pooling.SAGPool(nhid, ratio=pooling_ratio)

        self.lin1 = torch.nn.Linear(nhid * 2, nhid)
        self.lin2 = torch.nn.Linear(nhid, grph_dim)
        self.lin3 = torch.nn.Linear(grph_dim, label_dim)

        self.output_range = torch.nn.Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = torch.nn.Parameter(torch.FloatTensor([-3]), requires_grad=False)

        self.max_pool = dgl.nn.pytorch.glob.MaxPooling()
        self.avg_pool = dgl.nn.pytorch.glob.AvgPooling()

        if init_max:
            models.utils.init_max_weights(self)

    def forward(self, graph: dgl.DGLGraph):
        graph = utils.utils.add_batch_self_loop(graph)
        with graph.local_scope():
            
            # graph.ndata[FEATURES] = _feature_nan_fill(graph.ndata[FEATURES])
            # graph.ndata[FEATURES] = _feature_inf_limit(graph.ndata[FEATURES])
            graph.ndata[FEATURES] = _feature_normalize(graph.ndata[FEATURES])

            if self.position_encoding:
                graph = self.position_encoding(graph)

            x = graph.ndata[FEATURES]

            x = torch.nn.functional.relu(self.conv1(graph, x))
            graph, x, _ = self.pool1(graph, x)
            x1 = torch.cat([self.max_pool(graph, x), self.avg_pool(graph, x)], dim=1)

            graph = build_knn_graph(graph, self.dynamic_graph)
            x = torch.nn.functional.relu(self.conv2(graph, x))
            graph, x, _ = self.pool2(graph, x)
            x2 = torch.cat([self.max_pool(graph, x), self.avg_pool(graph, x)], dim=1)

            graph = build_knn_graph(graph, self.dynamic_graph)
            x = torch.nn.functional.relu(self.conv3(graph, x))
            graph, x, _ = self.pool3(graph, x)
            x3 = torch.cat([self.max_pool(graph, x), self.avg_pool(graph, x)], dim=1)

            x = x1 + x2 + x3

            x = torch.nn.functional.relu(self.lin1(x))
            x = torch.nn.functional.dropout(x, p=self.dropout_rate, training=self.training)
            features = torch.nn.functional.relu(self.lin2(x))
            out = self.lin3(features)
            if self.act is not None:
                out = self.act(out)

                if isinstance(self.act, torch.nn.Sigmoid):
                    out = out * self.output_range + self.output_shift

            return features, out

    def _creat_sage_shared_weights(self, in_dim, out_dim):
        out_weight = {'neigh_shared_weight': None, 'self_shared_weight': None}
        if self._weight_sharing:
            out_weight['neigh_shared_weight'] = torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
            out_weight['self_shared_weight'] = torch.nn.Linear(in_features=in_dim, out_features=out_dim, bias=False)
        return out_weight


def create_hetero_layer(module: torch.nn.Module, etypes=None, aggregate='sum', **kwargs):
    if etypes is None:
        return module(**kwargs)
    return dgl.nn.pytorch.HeteroGraphConv({e: module(**kwargs) for e in etypes}, aggregate=aggregate)


def _feature_fn_applier(x, fn):
    if isinstance(x, torch.Tensor):
        return fn(x)
    return {n: fn(f) for n, f in x.items()}


def _feature_inf_limit(x):
    def _tensor_limiter(_x):
        _x[_x == float('inf')] = 0
        _x[_x == -float('inf')] = 0
        return _x

    return _feature_fn_applier(x, _tensor_limiter)


def _feature_normalize(x):
    def _tensor_normalizer(_x):
        max_value = _x.max(0, keepdim=True)[0]
        max_value += (max_value == 0)
        _x = _x / max_value
        return _x

    return _feature_fn_applier(x, _tensor_normalizer)


def _feature_nan_fill(x):
    return _feature_fn_applier(x, torch.nan_to_num)


class MaxPooling(torch.nn.Module):
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, graph, feat, ntype=None):
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = dgl.max_nodes(graph, 'h', ntype=ntype)
            return readout


class AvgPooling(torch.nn.Module):
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, graph, feat, ntype=None):
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = dgl.mean_nodes(graph, 'h', ntype=ntype)
            return readout


class SharedSAGEConv(dgl.nn.pytorch.conv.SAGEConv):

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 neigh_shared_weight: torch.nn.Linear = None,
                 self_shared_weight: torch.nn.Linear = None,
                 **kwargs):
        super(SharedSAGEConv, self).__init__(in_feats, out_feats, aggregator_type, **kwargs)
        if neigh_shared_weight:
            if neigh_shared_weight.out_features == out_feats:
                self.fc_neigh = neigh_shared_weight
                print('Warning: share whole layer for shared SAGEConv')
            else:
                self.fc_neigh = torch.nn.Sequential(
                    neigh_shared_weight,
                    torch.nn.Linear(neigh_shared_weight.out_features, out_feats, bias=False)
                )
        if self_shared_weight:
            if self_shared_weight.out_features == out_feats:
                self.fc_self = self_shared_weight
                print('Warning: share whole layer for shared SAGEConv')
            else:
                self.fc_self = torch.nn.Sequential(
                    self_shared_weight,
                    torch.nn.Linear(self_shared_weight.out_features, out_feats, bias=False)
                )


class ResidualSAGEConv(dgl.nn.pytorch.conv.SAGEConv):

    def __init__(self, *args, **kwargs):
        super(ResidualSAGEConv, self).__init__(*args, activation=torch.relu, **kwargs)
        self.new_fc_self = torch.nn.Linear(self.fc_self.in_features, self.fc_self.out_features)

    def forward(self, graph, feat, edge_weight=None):
        torch.nn.init.zeros_(self.fc_self.weight)
        self.fc_self.weight.requires_grad = False
        return self.new_fc_self(feat) + super().forward(graph, feat, edge_weight)


class HeteroSAGEConv(dgl.nn.pytorch.SAGEConv):

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None,
                 neigh_shared_weight: torch.nn.Linear = None,
                 self_shared_weight: torch.nn.Linear = None,
                 node_type_count=None,
                 weight_sharing=False):
        super(HeteroSAGEConv, self).__init__(in_feats, out_feats, aggregator_type, feat_drop=feat_drop,
                                             bias=bias, norm=norm, activation=activation)
        edge_types = utils.utils.generate_edge_types(node_type_count)
        node_types = utils.utils.generate_node_types(node_type_count)
        self.self_shared_weight = self_shared_weight
        self.fc_self = dgl.nn.pytorch.TypedLinear(self_shared_weight.out_features, out_feats, len(node_types))
        # self.fc_self = dgl.nn.pytorch.TypedLinear(self._in_dst_feats, out_feats, len(node_types))
        # self.fc_self = self.fc_self
        if weight_sharing:
            self.fc_neigh = torch.nn.Sequential(
                neigh_shared_weight,
                torch.nn.Linear(neigh_shared_weight.out_features, self._in_src_feats, bias=False),
                # torch.nn.Linear(self._in_src_feats, self._in_src_feats, bias=False),
                torch.nn.BatchNorm1d(self._in_src_feats),
                torch.nn.ReLU()
            )
            # self.fc_neigh = self.fc_neigh
            self.hetero_fc = dgl.nn.pytorch.TypedLinear(self._in_src_feats, out_feats, len(edge_types))
            # self.hetero_fc = None
        else:
            self.fc_neigh = torch.nn.ModuleDict(
                {e: torch.nn.Linear(self._in_src_feats, out_feats, bias=False) for e in edge_types})

    def forward(self, graph, feat, edge_weight=None):
        r"""

        Description
        -----------
        Compute GraphSAGE layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N_{dst}, D_{out})`
            where :math:`N_{dst}` is the number of destination nodes in the input graph,
            :math:`D_{out}` is the size of the output feature.
        """
        self._compatibility_check()
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            msg_fn = dgl.function.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                msg_fn = dgl.function.u_mul_e('h', '_edge_weight', 'm')

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            # lin_before_mp = (self._in_src_feats > self._out_feats) or isinstance(self.fc_neigh, torch.nn.ModuleDict)
            lin_before_mp = True

            # Message Passing
            if self._aggre_type == 'mean':
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if self.hetero_fc is not None:
                    msg_fn = self.message_func
                graph.update_all(msg_fn, dgl.function.mean('m', 'neigh'))
                h_neigh = graph.dstdata['neigh']
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'gcn':
                dgl.utils.check_eq_shape(feat)
                graph.srcdata['h'] = self.fc_neigh(feat_src) if lin_before_mp else feat_src
                if isinstance(feat, tuple):  # heterogeneous
                    graph.dstdata['h'] = self.fc_neigh(feat_dst) if lin_before_mp else feat_dst
                else:
                    graph.dstdata['h'] = graph.srcdata['h']
                graph.update_all(msg_fn, dgl.function.sum('m', 'neigh'))
                # divide in_degrees
                degs = graph.in_degrees().to(feat_dst)
                h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                if not lin_before_mp:
                    h_neigh = self.fc_neigh(h_neigh)
            elif self._aggre_type == 'pool':
                graph.srcdata['h'] = torch.nn.functional.relu(self.fc_pool(feat_src))
                graph.update_all(msg_fn, dgl.function.max('m', 'neigh'))
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            elif self._aggre_type == 'lstm':
                graph.srcdata['h'] = feat_src
                graph.update_all(msg_fn, self._lstm_reducer)
                h_neigh = self.fc_neigh(graph.dstdata['neigh'])
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = h_neigh
            else:
                if isinstance(self.fc_self, dgl.nn.pytorch.TypedLinear):
                    rst = self.fc_self(self.self_shared_weight(h_self), graph.dstdata[TYPE_FIELD]) + h_neigh
                else:
                    rst = self.fc_self(h_self) + h_neigh

            # bias term
            if self.bias is not None:
                rst = rst + self.bias

            # activation
            if self.activation is not None:
                rst = self.activation(rst)
            # normalization
            if self.norm is not None:
                rst = self.norm(rst)
            return rst

    def message_func(self, edges):
        return {'m': self.hetero_fc(edges.src['h'], edges.data[TYPE_FIELD])}


class NoSelfSAGEConv(dgl.nn.pytorch.conv.SAGEConv):

    def __init__(self, *args, **kwargs):
        super(NoSelfSAGEConv, self).__init__(*args, **kwargs)

    def forward(self, graph, feat, edge_weight=None):
        torch.nn.init.zeros_(self.fc_self.weight)
        self.fc_self.weight.requires_grad = False
        return super().forward(graph, feat, edge_weight)        


class ConcatEdgeConv(dgl.nn.pytorch.conv.EdgeConv):

    def __init__(self, in_feat, out_feat, *args, **kwargs):
        super().__init__(in_feat, out_feat, *args, **kwargs)
        self.theta = torch.nn.Linear(2*in_feat, out_feat)
    
    def forward(self, g, feat):
        """

        Description
        -----------
        Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : Tensor or pair of tensors
            :math:`(N, D)` where :math:`N` is the number of nodes and
            :math:`D` is the number of feature dimensions.

            If a pair of tensors is given, the graph must be a uni-bipartite graph
            with only one edge type, and the two tensors must have the same
            dimensionality on all except the first axis.

        Returns
        -------
        torch.Tensor
            New node features.

        Raises
        ------
        DGLError
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.
        """
        with g.local_scope():
            if not self._allow_zero_in_degree:
                if (g.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            h_src, h_dst = expand_as_pair(feat, g)
            g.srcdata['x'] = h_src
            g.dstdata['x'] = h_dst

            ## my change
            def local_message(edges):
                return {'theta': self.theta(torch.cat((edges.src['x'], edges.dst['x']), dim=-1))}
            g.apply_edges(local_message)
            ## original implementation
            # g.apply_edges(dgl.function.v_sub_u('x', 'x', 'theta'))
            # g.edata['theta'] = self.theta(g.edata['theta'])

            g.dstdata['phi'] = self.phi(g.dstdata['x'])
            if not self.batch_norm:
                g.update_all(dgl.function.e_add_v('theta', 'phi', 'e'), dgl.function.max('e', 'x'))
            else:
                g.apply_edges(dgl.function.e_add_v('theta', 'phi', 'e'))
                # Although the official implementation includes a per-edge
                # batch norm within EdgeConv, I choose to replace it with a
                # global batch norm for a number of reasons:
                #
                # (1) When the point clouds within each batch do not have the
                #     same number of points, batch norm would not work.
                #
                # (2) Even if the point clouds always have the same number of
                #     points, the points may as well be shuffled even with the
                #     same (type of) object (and the official implementation
                #     *does* shuffle the points of the same example for each
                #     epoch).
                #
                #     For example, the first point of a point cloud of an
                #     airplane does not always necessarily reside at its nose.
                #
                #     In this case, the learned statistics of each position
                #     by batch norm is not as meaningful as those learned from
                #     images.
                g.edata['e'] = self.bn(g.edata['e'])
                g.update_all(dgl.function.copy_e('e', 'e'), dgl.function.max('e', 'x'))
            return g.dstdata['x']


class GraphTransformerLayer(torch.nn.Module):

    def __init__(self, hidden_feat, n_head, activation=torch.relu) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(hidden_feat)
        self.norm2 = torch.nn.LayerNorm(hidden_feat)
        self.gnn = torch.nn.ModuleList([dgl.nn.pytorch.SAGEConv(hidden_feat, hidden_feat, aggregator_type='mean') for _ in range(n_head)])
        self.gnn_fc = torch.nn.Linear(n_head * hidden_feat, hidden_feat)
        self.layer1 = torch.nn.Linear(hidden_feat, hidden_feat)
        self.dropout = torch.nn.Dropout(0.2)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.layer2 = torch.nn.Linear(hidden_feat, hidden_feat)
        self.activation = activation
        
    def forward(self, graph, x):
        x = x + self._gnn_block(graph, self.norm1(x))
        x = x + self._ffd_block(self.norm2(x))
        return x
    
    def _gnn_block(self, graph, x):
        x = torch.cat([m(graph, x) for m in self.gnn], dim=1)
        x = self.gnn_fc(self.activation(x))
        return self.dropout1(x)

    def _ffd_block(self, x):
        x = self.layer2(self.dropout(self.activation(self.layer1(x))))
        return self.dropout2(x)


class GNNTransformer(torch.nn.Module):

    def __init__(self, in_feat, hidden_feat, n_head, n_layer, position_encoding, dynamic_graph, graph_dim=32, activation=torch.relu) -> None:
        super().__init__()

        self.dynamic_graph = dynamic_graph
        self.preconv = torch.nn.Sequential(torch.nn.Linear(in_feat, hidden_feat), torch.nn.BatchNorm1d(hidden_feat), torch.nn.ReLU(), torch.nn.Linear(hidden_feat, hidden_feat))
        self.gnns = torch.nn.ModuleList([GraphTransformerLayer(hidden_feat, n_head, activation) for i in range(n_layer)])
        self.pools = torch.nn.ModuleList([models.pooling.SAGPool(hidden_feat) for _ in range(n_layer)])
        self.positions = torch.nn.ModuleList([get_position_encoding(position_encoding if i == 0 else None, hidden_feat, hidden_feat)[0] for i in range(n_layer)])

        self.fc = torch.nn.Sequential(torch.nn.Linear(hidden_feat * 2, hidden_feat), torch.nn.ReLU(), torch.nn.Dropout(0.2), torch.nn.Linear(hidden_feat, graph_dim), torch.nn.ReLU())

        self.max_pool = dgl.nn.pytorch.glob.MaxPooling()
        self.avg_pool = dgl.nn.pytorch.glob.AvgPooling()

    def forward(self, graph):
        with graph.local_scope():
            graph = utils.utils.add_batch_self_loop(graph)

            x = self.preconv(graph.ndata[FEATURES])

            picks = []
            for i in range(len(self.gnns)):
                graph = build_knn_graph(graph, self.dynamic_graph)
                x = self.positions[i](graph, x)
                x = self.gnns[i](graph, x)
                graph, x, _ = self.pools[i](graph, x)
                picks.append(torch.cat([self.max_pool(graph, x), self.avg_pool(graph, x)], dim=1))
            
            features = self.fc(sum(picks))

            return features, None  # no need for point prediction here


class SimpleGatedAttention(torch.nn.Module):
    def __init__(self, in_feat = 1024, nhid = 256, norm=None, dropout = True, n_classes = 1, condition_enable=False, mutual_attention=False):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super().__init__()
        self.attention = torch.nn.Sequential(torch.nn.Linear(in_feat, nhid))
        self.cls = torch.nn.Sequential(torch.nn.GELU(), torch.nn.Dropout(0.25), torch.nn.Linear(nhid, n_classes))
        
        self.mutual_attention = mutual_attention
        if condition_enable:
            if not self.mutual_attention:
                self.condition_network = torch.nn.Sequential(torch.nn.Linear(nhid, 1), torch.nn.Sigmoid())
            else:
                self.condition_network = torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(nhid, nhid))

    def forward(self, graph, x, condition=None):
        buttleneck = self.attention(x)
        A = self.cls(buttleneck)
        
        if condition is not None:
            condition_attn = self.condition_network(condition) # n_graph_batch * nhid
            condition_attn = torch.repeat_interleave(condition_attn, graph.batch_num_nodes(), dim=0) # n_nodes * nhid
            if not self.mutual_attention:
                A = A * condition_attn
            else:
                A = A * torch.sigmoid((buttleneck * condition_attn).sum(dim=-1, keepdim=True))

        # print('shapes ', A.shape, x.shape)
        idx = 0
        output = []
        for n in graph.batch_num_nodes():
            output.append(torch.sum(torch.nn.functional.softmax(A[idx:idx+n, ...], dim=0) * x[idx:idx+n:, ...], dim=0, keepdim=True))
            idx += n
        return torch.cat(output, dim=0)


class GatedAttention(torch.nn.Module):
    def __init__(self, in_feat = 1024, nhid = 256, norm=None, dropout = True, n_classes = 1, condition_enable=False, mutual_attention=False):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(GatedAttention, self).__init__()
        self.attention_a = [
            torch.nn.Linear(in_feat, nhid),
            torch.nn.Tanh()]
        
        self.attention_b = [torch.nn.Linear(in_feat, nhid), torch.nn.Sigmoid()]
        if dropout:
            self.attention_a.append(torch.nn.Dropout(0.25))
            self.attention_b.append(torch.nn.Dropout(0.25))

        self.attention_a = torch.nn.Sequential(*self.attention_a)
        self.attention_b = torch.nn.Sequential(*self.attention_b)
        self.attention_c = torch.nn.Linear(nhid, n_classes)

        if condition_enable:
            if not mutual_attention:
                self.condition_network = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.Sigmoid())
            else:
                raise NotImplementedError('needs the buttleneck implementation')
                self.condition_network = torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(nhid, nhid))
                self.buttleneck = torch.nn.Linear(in_feat, nhid)

    def forward(self, graph, x, condition=None):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)

        if condition is not None:
            condition_attn = self.condition_network(condition) # n_graph_batch * nhid
            condition_attn = torch.repeat_interleave(condition_attn, graph.batch_num_nodes(), dim=0) # n_nodes * nhid
            if not self.mutual_attention:
                A = A.mul(condition_attn)
            else:
                A = A.mul(torch.sigmoid((self.buttleneck(x) * condition_attn).sum(dim=-1)))

        A = self.attention_c(A)  # N x n_classes
        A_path = torch.transpose(A, 1, 0) # n_classes x N
        idx = 0
        output = []
        for n in graph.batch_num_nodes():
            output.append(torch.mm(torch.nn.functional.softmax(A_path[..., idx:idx+n], dim=1), x[idx:idx+n:, ...]))
            idx += n
        return torch.cat(output, dim=0)


class GatedVarAttention(torch.nn.Module):
    def __init__(self, in_feat = 1024, nhid = 256, norm=None, dropout=True, n_head=1, n_classes=1, condition_enable=False, mutual_attention=False):
        r"""
        Attention Network with Sigmoid Gating (3 fc layers)

        args:
            L (int): input feature dimension
            D (int): hidden layer dimension
            dropout (bool): whether to apply dropout (p = 0.25)
            n_classes (int): number of classes
        """
        super(GatedVarAttention, self).__init__()
        self.heads = torch.nn.ModuleList([self.GatedVarHead(in_feat, nhid, norm, dropout, n_classes, condition_enable=condition_enable, mutual_attention=mutual_attention) for _ in range(n_head)])
        self.mlp = torch.nn.Sequential(torch.nn.Linear(n_head*nhid*2, n_head*nhid), torch.nn.ReLU(), torch.nn.Linear(n_head*nhid, nhid))

    def forward(self, graph, x, condition=None):
        output = []
        for i in range(len(self.heads)):
            output.append(self.heads[i](graph, x, condition))
        return self.mlp(torch.cat(output, dim=1))
    
    class GatedVarHead(torch.nn.Module):
        def __init__(self, in_feat = 1024, nhid = 256, norm=None, dropout = True, n_head=1, n_classes = 1, condition_enable=False, mutual_attention=False):
            super().__init__()

            self.attention_a = [
                torch.nn.Linear(in_feat, nhid),
                torch.nn.Tanh()]
            
            self.attention_b = [torch.nn.Linear(in_feat, nhid), torch.nn.Sigmoid()]
            if dropout:
                self.attention_a.append(torch.nn.Dropout(0.25))
                self.attention_b.append(torch.nn.Dropout(0.25))

            self.attention_a = torch.nn.Sequential(*self.attention_a)
            self.attention_b = torch.nn.Sequential(*self.attention_b)
            self.attention_c = torch.nn.Linear(nhid, n_classes)

            if condition_enable:
                if not mutual_attention:
                    self.condition_network = torch.nn.Sequential(torch.nn.Linear(nhid, nhid), torch.nn.Sigmoid())
                else:
                    raise NotImplementedError('needs the buttleneck implementation')
                    self.condition_network = torch.nn.Sequential(torch.nn.GELU(), torch.nn.Linear(nhid, nhid))
        
        def forward(self, graph, x, condition=None):
            a = self.attention_a(x) # n_nodes * nhid
            b = self.attention_b(x) # n_nodes * nhid
            A = a.mul(b) # n_nodes * nhid

            if condition is not None:
                condition_attn = self.condition_network(condition) # n_graph_batch * nhid
                condition_attn = torch.repeat_interleave(condition_attn, graph.batch_num_nodes(), dim=0) # n_nodes * nhid
                if not self.mutual_attention:
                    A = A.mul(condition_attn)
                else:
                    A = A.mul(torch.sigmoid((A * condition_attn).sum(dim=-1)))

            A = self.attention_c(A)  # n_nodes x 1
                
            A_path = torch.transpose(A, 1, 0) # n_classes x N
            idx = 0
            output = []
            for n in graph.batch_num_nodes():
                attn = torch.nn.functional.softmax(A_path[..., idx:idx+n], dim=1)

                weighted_sum = torch.mm(attn, x[idx:idx+n:, ...])

                output.append(torch.cat((weighted_sum, torch.mm(attn, (x[idx:idx+n:, ...] - weighted_sum)**2)), dim=-1))
                idx += n
            return torch.cat(output, dim=0)


class SimilarityExtractor(models.pooling.SimilarityScorer):

    def __init__(self, out_feat) -> None:
        super().__init__()
        self.mlp = torch.nn.Sequential(torch.nn.Linear(1, out_feat//2), torch.nn.BatchNorm1d(out_feat//2), torch.nn.ReLU(), torch.nn.Linear(out_feat//2, out_feat))
    
    def forward(self, graph, feature):
        x = super().forward(graph, feature)
        return torch.cat((feature, self.mlp(x.unsqueeze(-1))), dim=1)


def get_similarity_encoding(similarity_encoding, in_feat, out_feat):
    if not similarity_encoding:
        return NonePositinEncoding(), in_feat
    return SimilarityExtractor(out_feat=out_feat-in_feat), out_feat
