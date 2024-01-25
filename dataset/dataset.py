from cProfile import run
import copy
import mmap
import os
import pickle
import typing

import dgl
import dgl.data.utils
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
from tqdm import tqdm

import dataset
import utils.graph
import utils.utils
from data_workflow.histocartography_extension.graph_builder import TYPE as TYPE_FIELD
from data_workflow.meta_data import meta_data, folding

DATA_SAVE_KEY = 'DATA'


class GraphDataset(dgl.data.DGLDataset):

    def __init__(self, name: str, raw_dir: str = None, show_bar: bool = False, transforms=None, runtime_transforms=None,
                 cache=True, save_dir: str = None, subtypes: str = None, distance_feature: bool = False,
                 bin_count=4, hetero_type_count=None, dataset_name=None):
        assert dataset_name is not None
        self._dataset_name = dataset_name
        self._core_to_study_id = None
        self._outcome = None
        self._distance_feature = distance_feature
        self.show_bar = show_bar
        self._data_files = []  # list of tuples (Graph, censored, partial, time, study_id, subtype)
        self._study_ids = None
        self._censor_status = None
        self._transforms = transforms
        if runtime_transforms is not None and not isinstance(runtime_transforms, list):
            runtime_transforms = [runtime_transforms]
        self._runtime_transforms = runtime_transforms
        self.type_map = None
        if hetero_type_count:
            edge_types = utils.utils.generate_edge_types(hetero_type_count)
            self.type_map = {e: i for i, e in enumerate(edge_types)}
        self._invalid_study_ids = []
        if self._transforms is not None and not isinstance(self._transforms, list):
            self._transforms = [self._transforms]
            name += f'_trans{len(self._transforms)}'
        self._cache = cache
        self._subtypes = [s.lower() for s in subtypes] if subtypes is not None else None
        if self._subtypes:
            name += '_' + '_'.join(self._subtypes) + '_' + f'distance_{self._distance_feature}'
        self._bin_count = bin_count
        name += f'_bin{self._bin_count}'
        os.makedirs(save_dir, exist_ok=True)
        # name += '.tar'  # add suffix
        super(GraphDataset, self).__init__(name=name, raw_dir=raw_dir, save_dir=save_dir)

    def save(self):
        if self.name is None or self.save_dir is None:
            return
        for i, d in enumerate(self._data_files):
            with open(os.path.join(self.save_dir, self.name + f'_{i}.pkl'), 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(self.save_dir, self.name + '_summary.pkl'), 'wb') as handle:
            pickle.dump((self._study_ids, self._censor_status, self.file_name), handle, protocol=pickle.HIGHEST_PROTOCOL)
        # dgl.data.utils.save_info(os.path.join(self.save_dir, self.name), {DATA_SAVE_KEY: self._data_files})

    def load(self):
        with open(os.path.join(self.save_dir, self.name + '_summary.pkl'), 'rb') as handle:
            self._study_ids, self._censor_status, self.file_name = pickle.load(handle)
        # self._data_files = dgl.data.utils.load_info(os.path.join(self.save_dir, self.name))[DATA_SAVE_KEY]
        # pass

    def has_cache(self):
        return os.path.exists(os.path.join(self.save_dir, self.name + '_summary.pkl'))

    def process(self):
        """
        Read graphs from the data path with their corresponding study IDs
        """
        self._outcome = pd.read_csv(os.path.join(self.raw_dir, dataset.const.OUTCOME_FILENAME))
        self.add_interval_field()

        self._core_to_study_id = pd.read_csv(os.path.join(self.raw_dir, dataset.const.CORE_ID_FILENAME))
        self._invalid_study_ids = pd.read_csv(
            os.path.join(self.raw_dir, dataset.const.INVALID_STUDY_IDS_FILENAME)).study_id.values
        graph_files = utils.utils.list_files(self.raw_dir, '.bin')

        self.file_name = []
        for f in tqdm(graph_files, disable=not self.show_bar):
            data, is_valid = self._read_record(f)
            if not is_valid:
                continue
            if not self._cache:
                if self._distance_feature:
                    raise NotImplemented('distance feature not implemented without cache')
                self._data_files.append(f)
                continue
            if self._distance_feature:
                data = (utils.graph.concat_edge_distance_features(data[0]),) + data[1:]
            self._data_files.append(data)
            self.file_name.append(os.path.split(f)[1])
        self._study_ids = [d[-1] for d in self._data_files]
        self._censor_status = [d[1] for d in self._data_files]

    def add_interval_field(self):
        disc_labels, q_bins = pd.qcut(self._outcome.loc[self._outcome.status != dataset.const.CENSORED_STATUS, 'time'],
                                      q=self._bin_count, retbins=True, labels=False)
        eps = 1e-6
        q_bins[-1] = self._outcome['time'].max() + eps
        q_bins[0] = self._outcome['time'].min() - eps

        disc_labels, q_bins = pd.cut(self._outcome['time'], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        self._outcome['interval'] = disc_labels.values.astype(int)

    def _read_record(self, record_path: str):
        graph = dgl.load_graphs(record_path)

        # core_id = graph[1]['patient_id'].item()  # field name subject to change to "core_id"
        core_id = meta_data.filename_to_patient_id(record_path, self._dataset_name)
        study_id = self._core_to_study_id.loc[self._core_to_study_id.core_id == core_id]
        if study_id.shape[0] == 0:
            return None, False
        assert study_id.shape[0] == 1
        study_id = study_id.study_id.values[0]

        if study_id in self._invalid_study_ids:
            return None, False

        patient_outcome = self._outcome.loc[self._outcome.study_id == study_id]
        if patient_outcome.shape[0] == 0:  # have to skip the patients without known status
            return None, False

        censored = patient_outcome.status.values[0] == dataset.const.CENSORED_STATUS
        partial = patient_outcome.status.values[0] == dataset.const.PARTIAL_STATUS
        time = patient_outcome.time.values[0]
        subtype = patient_outcome.subtype.values[0]
        interval = patient_outcome.interval.values[0]

        if self._subtypes is not None and subtype.lower() not in self._subtypes:
            return None, False

        if partial:
            return None, False

        graph = graph[0][0]
        # todo: might not be a correct assumption : remove empty nodes
        # graph = utils.remove_isolated_nodes(graph)

        if self._transforms is not None:
            for transform in self._transforms:
                graph = transform(graph)

        self.add_edge_type(graph)
        return (graph, censored, partial, time, interval, study_id), True

    def __len__(self):
        # return len(self._data_files)
        return len(self._study_ids)

    def __getitem__(self, item):
        record = self._read_h5(item)
        # if not self._cache:
        #     record = self._read_record(record)  # [0], it was here, but I think it is wrong
        if self._runtime_transforms:
            record = copy.deepcopy(record)
            for transform in self._runtime_transforms:
                record = (transform(record[0]),) + record[1:]
        return record

    def _read_h5(self, index):
        if index >= self.__len__():
            raise IndexError('index out of bound')
        with open(os.path.join(self.save_dir, self.name + f'_{index}.pkl'), 'rb') as handle:
            with mmap.mmap(handle.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_file_obj:
                return pickle.load(mmap_file_obj)
            # return pickle.load(handle)

    def stats(self, obj):
        rep = f"dataset size: {obj.__len__()}\n"

        df = pd.DataFrame({'censor': obj.censor_status, 'study_id': obj.study_ids})

        rep += f"patient count: {len(df.study_id.unique())}\n"
        rep += f"censor count: {len(df.loc[df.censor == True].study_id.unique())}\n"

        rep += f"study ids: {df.study_id.values}\n"

        return rep

    def get_file_name(self, index):
        return self.file_name[index]

    @property
    def study_ids(self):
        return self._study_ids

    @property
    def censor_status(self):
        return self._censor_status

    def __repr__(self):
        return self.stats(self)

    def add_edge_type(self, graph):
        if self.type_map is None:
            return

        def edge_func(edges):
            etypes = torch.Tensor([self.type_map[f'{min(x, y)}_{max(x, y)}'] for x, y in
                                   zip(edges.src[TYPE_FIELD], edges.dst[TYPE_FIELD])])
            return {TYPE_FIELD: etypes}

        graph.apply_edges(edge_func)

    @property
    def dataset_name(self):
        return self._dataset_name


class GraphSubDataset(dgl.data.utils.Subset):

    def __init__(self, dt: GraphDataset, ids: typing.List[int], include: bool = True,
                 censor_portion: float = None) -> dgl.data.utils.Subset:
        """
        Selects a subset of the Graph Dataset based on study IDs
        Args:
            dt (GraphDataset): input dataset
            ids (List[int]): study ids to select
            include (bool): whether "ids" are included or want to be excluded
            censor_portion (float): portion of the cases to be used as censor
        Return:
            (dgl.data.utils.Subset) a dataset with the selected study IDs
        """

        self.transform = None
        self.censor_portion = censor_portion
        df = pd.DataFrame({'study_id': dt.study_ids, 'censor': dt.censor_status})
        dataset_idx = df.loc[df.study_id.isin(ids)].index.values
        # dataset_idx = [i for i, study_id in enumerate(dt.study_ids) if study_id in ids]
        if not include:
            dataset_idx = df.loc[~df.study_id.isin(ids)].index.values
            # dataset_idx = [i for i in range(len(dt)) if i not in dataset_idx]
        df = df.loc[dataset_idx].reset_index(drop=True)
        # status = df.loc[df.censor == True].study_id.values
        # status = [censor for i, censor in enumerate(dt.censor_status) if i in dataset_idx]
        self.indexes = {
            # 'censor': [i for i, c in enumerate(status) if c],
            # 'event': [i for i, c in enumerate(status) if not c]
            'censor': df.loc[df.censor == True].index.values,
            'event': df.loc[df.censor == False].index.values
        }
        super(GraphSubDataset, self).__init__(dt, dataset_idx)

    def set_transform(self, transform):
        if transform is not None and not isinstance(transform, list):
            transform = [transform]
        self.transform = transform

    def set_censor_portion(self, portion):
        self.censor_portion = portion

    def get_file_name(self, index):
        return self.dataset.get_file_name(self.indices[index])

    @property
    def study_ids(self):
        df = pd.DataFrame({'study_id': self.dataset.study_ids, 'censor': self.dataset.censor_status})
        df = df.loc[self.indices]
        return df.study_id.values
        # return [s for i, s in enumerate(self.dataset.study_ids) if i in self.indices]

    @property
    def censor_status(self):
        df = pd.DataFrame({'study_id': self.dataset.study_ids, 'censor': self.dataset.censor_status})
        df = df.loc[self.indices]
        return df.censor.values
        # return [s for i, s in enumerate(self.dataset.censor_status) if i in self.indices]

    def __getitem__(self, item):
        if item >= self.__len__():
            raise IndexError('Index out of range error')
        if self.censor_portion is not None:
            index = self.indexes['event']
            p = np.random.uniform()
            if p < self.censor_portion:
                index = self.indexes['censor']
            item = index[item % len(index)]
        data = super().__getitem__(item)
        if self.transform:
            data = copy.deepcopy(data)
            for transform in self.transform:
                data = (transform(data[0]),) + data[1:]
        return data

    def __repr__(self):
        # make sensor portion null to go over indexes sequentially
        censor_portion = self.censor_portion
        self.censor_portion = None
        rep = self.dataset.stats(self) + f"censor portion: {self.censor_portion}\n"
        self.censor_portion = censor_portion
        return rep


def split_dataset(dg: GraphDataset, split_study_ids: typing.List[int]) -> (GraphDataset, GraphDataset):
    """
    Splits the dataset into two dataset including and excluding the study ID lists
    Args:
        dg (GraphDataset): graph dataset to break
        split_study_ids (List[int]): study ID list
    Return:
        (GraphDataset, GraphDataset): include and exclude datasets
    """
    include_set = GraphSubDataset(dg, split_study_ids, include=True)
    exclude_set = GraphSubDataset(dg, split_study_ids, include=False)
    return include_set, exclude_set


def k_fold_split_dataset(dg: GraphDataset, k: int, censor_portion: float = None) -> typing.List[typing.Tuple]:
    """
    Splits dataset into K-folds based on study ids
    Args:
        dg (GraphDataset): graph dataset
        k (int): number of folds
        censor_portion (float): portion of censor cases in a batch
    Return:
        List[(GraphDataset, GraphDataset)]: list of train/test datasets
    """
    study_ids = np.array(dg.study_ids)
    graphs = []
    for train_study_ids, test_study_ids in folding.get_folded_study_ids(study_ids, k, dg.dataset_name):
        train_graph = GraphSubDataset(dg, train_study_ids, censor_portion=censor_portion)
        test_graph = GraphSubDataset(dg, test_study_ids)
        graphs.append((train_graph, test_graph))
    return graphs
