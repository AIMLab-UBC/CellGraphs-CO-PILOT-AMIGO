import argparse
import collections
import copy
import datetime
import itertools
import os
import random
import time

import dgl
import lifelines.statistics
import lifelines.utils
import numpy as np
import pandas as pd
import scipy.sparse
import torch
import torch.distributed


def list_files(directory, ext=None):
    return [os.path.join(directory, d) for d in os.listdir(directory) if ext is None or d.endswith(ext)]


def set_seeds(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def nullable_int_flag(s):
    """
    Parse int with making understanding None
    """
    s = s.lower()
    if s == 'none':
        return None
    s = int(s)
    return s


def nullable_float_flag(s):
    s = s.lower()
    if s == 'none':
        return None
    s = float(s)
    return s


def nullable_str_flag(s):
    """
    Parse int with making understanding None
    """
    if s.lower() == 'none':
        return None
    return s.lower()


def nullable_str_flag_none_lower(s):
    """
    Parse int with making understanding None
    """
    if s.lower() == 'none':
        return None
    return s


def is_dist_avail_and_initialized():
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.6f} ({global_avg:.6f})"
        self.deque = collections.deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(median=self.median, avg=self.avg, global_avg=self.global_avg, max=self.max,
                               value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = collections.defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, total_length=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.6f}')
        data_time = SmoothedValue(fmt='{avg:.6f}')
        if total_length is None:
            total_length = len(iterable)
        space_fmt = ':' + str(len(str(total_length))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}',
                 'max mem: {memory:.0f}'])
        else:
            log_msg = self.delimiter.join(
                [header, '[{0' + space_fmt + '}/{1}]', 'eta: {eta}', '{meters}', 'time: {time}', 'data: {data}'])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total_length - 1:
                eta_seconds = iter_time.global_avg * (total_length - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(i, total_length, eta=eta_string, meters=str(self), time=str(iter_time),
                                         data=str(data_time), memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(i, total_length, eta=eta_string, meters=str(self), time=str(iter_time),
                                         data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.6f} s / it)'.format(header, total_time_str, total_time / total_length))


def c_index_score(hazards, censored, survtime_all):
    try:
        return lifelines.utils.concordance_index(survtime_all, -hazards, 1 - censored)
    except:
        return 0.5


def cox_log_rank(hazardsdata, censored, survtime_all):
    labels = 1 - censored
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = lifelines.statistics.logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return pvalue_pred


def accuracy_cox(hazardsdata, censored):
    labels = 1 - censored
    # This accuracy is based on estimated survival events against true survival events
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def make_csr_matrix_symmetric(matrix: scipy.sparse.csr_matrix) -> scipy.sparse.csr_matrix:
    matrix = matrix.copy()
    rows, cols = matrix.nonzero()
    matrix[cols, rows] = matrix[rows, cols]
    return matrix


def is_csr_matrix_symmetric(matrix: scipy.sparse.csr_matrix) -> bool:
    return len((matrix - matrix.T).nonzero()[0]) == 0


def hazard_average_by_patient(hazard: np.ndarray, censor: np.ndarray, partial: np.ndarray, surv_time: np.ndarray,
                              study_id: np.ndarray, drop_partial=False):
    df = pd.DataFrame(
        {'hazard': hazard, 'censor': censor, 'partial': partial, 'surv_time': surv_time, 'study_id': study_id})

    if drop_partial:
        df = df.loc[df.partial == 0]

    aggregation = df.groupby('study_id').agg('max')
    aggregation['study_id'] = aggregation.index  # to get the study id as well
    return aggregation.hazard.values, aggregation.censor.values, aggregation.partial.values, aggregation.surv_time.values, aggregation


def remove_isolated_nodes(graph: dgl.DGLGraph):
    isolated_nodes = ((graph.in_degrees() == 0) & (graph.out_degrees() == 0)).nonzero().squeeze(1)
    graph.remove_nodes(isolated_nodes)
    return graph


def get_batch_info(graph: dgl.DGLGraph):
    etypes = graph.canonical_etypes
    ntypes = graph.ntypes
    batch_num_nodes = {n: graph.batch_num_nodes(ntype=n) for n in ntypes} if ntypes else graph.batch_num_nodes()
    batch_num_edges = {e: graph.batch_num_edges(etype=e) for e in etypes} if etypes else graph.batch_num_edges()
    return batch_num_nodes, batch_num_edges


def set_batch_info(graph: dgl.DGLGraph, batch_num_nodes, batch_num_edges):
    if batch_num_nodes is not None:
        graph.set_batch_num_nodes(batch_num_nodes)
    if batch_num_edges is not None:
        graph.set_batch_num_edges(batch_num_edges)
    return graph


def add_batch_self_loop(graph: dgl.DGLGraph):
    """
    NOTE: this changes the order of the edges
    """
    assert len(graph.canonical_etypes) == 1, "hetero graph is not supported for now"

    # remove the current self-loops
    graph = dgl.remove_self_loop(graph)

    # keep batch info
    next_batch_nodes = graph.batch_num_nodes()
    next_batch_edges = graph.batch_num_edges() + next_batch_nodes

    # add self-loops
    graph = dgl.add_self_loop(graph)

    # reorder based on node to make sure self-loops are added to their corresponding batch
    graph = dgl.reorder_graph(graph, edge_permute_algo='src', store_ids=False)

    # set batch info
    graph.set_batch_num_nodes(next_batch_nodes)
    graph.set_batch_num_edges(next_batch_edges)

    check_batch_validity(graph)
    return graph


def batch_to_bidirected(graph, copy_ndata=True, manual=True):
    """
    NOTE: this changes the order of the edges
    """
    # get batch info
    num_nodes = graph.batch_num_nodes()
    num_edges = graph.batch_num_edges()

    # count number of added edges
    # adj = graph.adj(ctx=graph.device, scipy_fmt='csr')
    # d = adj + adj.transpose()
    # d[d == 2] = 0
    # d = d.sum(axis=1).squeeze(-1)
    # ndoes = torch.repeat_interleave(torch.arange(d.shape[0]), torch.from_numpy(d))

    # edges = {(x.item(), y.item()) for x, y in zip(*graph.edges())}
    # # reverse_edges = {(y, x) for x, y in edges}
    # reverse_edges = {(x.item(), y.item()) for x, y in zip(*graph.reverse().edges())}
    # nodes = [x[0] for x in set(reverse_edges) - set(edges)]
    # # edges = graph.edges()
    # # self_loop_nodes = edges[0][edges[0] == edges[1]]
    # batch_edges_bin = torch.nn.functional.pad(torch.cumsum(num_edges, dim=0), (1, 0), "constant", 0)
    # batch_self_loops = torch.histogram(torch.FloatTensor(nodes), batch_edges_bin.cpu().type(torch.FloatTensor))[0]
    # batch_self_loops = batch_self_loops.type(torch.LongTensor).to(graph.device)
    # # num_edges = num_edges + num_edges - batch_self_loops
    # num_edges = num_edges + batch_self_loops
    
    # to bidirected
    if not manual:
        graph = dgl.to_bidirected(graph.cpu(), copy_ndata=copy_ndata).to(graph.device)
    else:
        device = graph.device
        graph = copy.deepcopy(graph).to(device)
        adj = graph.adj(ctx=graph.device, scipy_fmt='csr')
        d = (adj > adj.transpose()).nonzero()
        graph.add_edges(torch.from_numpy(d[1]).type(torch.int64).to(device), torch.from_numpy(d[0]).type(torch.int64).to(device))

    # reorder based on node to make sure self-loops are added to their corresponding batch
    graph = dgl.reorder_graph(graph, edge_permute_algo='src', store_ids=False)

    # convert the edge numbers to node numbers
    batch_nodes_bin = torch.nn.functional.pad(torch.cumsum(num_nodes, dim=0), (1, 0), "constant", 0)
    batch_edge_count = torch.histogram(graph.edges()[0].cpu().type(torch.FloatTensor), batch_nodes_bin.cpu().type(torch.FloatTensor))[0]
    batch_edge_count = batch_edge_count.type(torch.LongTensor).to(graph.device)

    # set batch info
    graph.set_batch_num_nodes(num_nodes)
    graph.set_batch_num_edges(batch_edge_count)

    return graph

def check_batch_validity(graph):
    return
    with torch.no_grad():
        assert graph.num_nodes() == graph.batch_num_nodes().sum()
        assert graph.num_edges() == graph.batch_num_edges().sum()
        assert graph.batch_num_nodes().size(0) == graph.batch_num_edges().size(0)
        assert torch.equal(graph.nodes().cpu(), torch.arange(graph.num_nodes()))
        assert graph.edges()[0].max() < graph.num_nodes()
        assert graph.edges()[1].max() < graph.num_nodes()
        edge_cum_sum = torch.nn.functional.pad(torch.cumsum(graph.batch_num_edges(), dim=0), (1, 0), 'constant', 0)
        node_cum_sum = torch.nn.functional.pad(torch.cumsum(graph.batch_num_nodes(), dim=0), (1, 0), 'constant', 0)
        for batch_id in range(1, len(edge_cum_sum)):
             assert graph.edges()[0][edge_cum_sum[batch_id-1]:edge_cum_sum[batch_id]].max() < node_cum_sum[batch_id]


def generate_node_types(node_type_count):
    if not node_type_count:
        return None
    return [str(i) for i in range(node_type_count)]


def generate_edge_types(node_type_count):
    if not node_type_count:
        return None
    edge_type_names = []
    for i, comb in enumerate(itertools.combinations(list(range(node_type_count)), 2)):
        edge_type_names.append('{}_{}'.format(*comb))
    for i, node_type in enumerate(range(node_type_count), start=len(edge_type_names)):
        edge_type_names.append('{}_{}'.format(node_type, node_type))
    return edge_type_names


def generate_hetero_schema(node_type_count) -> dgl.heterograph:
    node_data = {n: 0 for n in generate_node_types(node_type_count)}
    edge_data = {}
    for g in generate_edge_types(node_type_count):
        edge_data[(g.split('_')[0], g, g.split('_')[1])] = []
        edge_data[(g.split('_')[1], g, g.split('_')[0])] = []
    return dgl.heterograph(edge_data, node_data)


def hetero_to_homo(graph: dgl.DGLHeteroGraph, x=None) -> dgl.DGLGraph:
    with graph.local_scope():
        if x is not None:
            for n in graph.ntypes:
                graph.nodes[n].data['x'] = x[n]
        batch_nodes = [graph.batch_num_nodes(n) for n in graph.ntypes]
        batch_edges = [graph.batch_num_edges(e) for e in graph.canonical_etypes]
        batch_nodes = torch.stack(batch_nodes, dim=0).sum(dim=0)
        batch_edges = torch.stack(batch_edges, dim=0).sum(dim=0)
        graph = dgl.to_homogeneous(graph, ndata=['x']).to(graph.device)
        graph.set_batch_num_nodes(batch_nodes)
        graph.set_batch_num_edges(batch_edges)
        if x is not None:
            return graph, graph.ndata['x']
        return graph


def homo_to_hetero(graph: dgl.DGLGraph, node_type_count,
                   ntype_field='type', etype_field='type') -> dgl.DGLHeteroGraph:
    if not node_type_count:
        return graph
    edge_type_to_type_id, edge_type_names = {}, []
    for i, comb in enumerate(itertools.combinations(list(range(node_type_count)), 2)):
        edge_type_to_type_id[comb] = i
        edge_type_names.append('{}_{}'.format(*comb))
    for i, node_type in enumerate(range(node_type_count), start=len(edge_type_names)):
        edge_type_to_type_id[(node_type, node_type)] = i
        edge_type_names.append('{}_{}'.format(node_type, node_type))

    with graph.local_scope():
        edge_type = []
        for src, dest in zip(*graph.edges()):
            src, dest = src.item(), dest.item()
            src_type, dest_type = graph.ndata[ntype_field][src].item(), graph.ndata[ntype_field][dest].item()
            key = (min(src_type, dest_type), max(src_type, dest_type))
            edge_type.append(edge_type_to_type_id[key])
        graph.edata[etype_field] = torch.LongTensor(edge_type).to(graph.device)
        schema = generate_hetero_schema(node_type_count)
        hetero_graph = dgl.to_heterogeneous(graph,
                                            ntypes=[str(i) for i in range(node_type_count)],
                                            etypes=edge_type_names,
                                            ntype_field=ntype_field, etype_field=etype_field,
                                            metagraph=schema.metagraph()).to(graph.device)
    return hetero_graph
