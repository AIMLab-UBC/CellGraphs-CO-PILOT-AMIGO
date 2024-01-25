import copy
import os
import typing

import dgl
import dgl.data.utils
import numpy as np
import pandas as pd
from tqdm import tqdm

import dataset
import utils.utils
from data_workflow.meta_data import meta_data, folding
from dataset.dataset import GraphDataset

DATA_SAVE_KEY = 'DATA'


class MultiModalGraphDataset(GraphDataset):

    def __init__(self, slide_types: str, **kwargs):
        self._slide_types = [s.lower() for s in set(slide_types)]
        super(MultiModalGraphDataset, self).__init__(**kwargs)
        assert self._cache

    @property
    def slide_types(self):
        return self._slide_types

    def process(self):
        """
        Read graphs from the data path with their corresponding study IDs
        """
        self._outcome = pd.read_csv(os.path.join(self.raw_dir, dataset.const.OUTCOME_FILENAME))

        ############################################################################################
        disc_labels, q_bins = pd.qcut(self._outcome.loc[self._outcome.status != dataset.const.CENSORED_STATUS, 'time'],
                                      q=self._bin_count, retbins=True, labels=False)
        eps = 1e-6
        q_bins[-1] = self._outcome['time'].max() + eps
        q_bins[0] = self._outcome['time'].min() - eps

        disc_labels, q_bins = pd.cut(self._outcome['time'], bins=q_bins, retbins=True, labels=False, right=False,
                                     include_lowest=True)
        self._outcome['interval'] = disc_labels.values.astype(int)
        ##################################################################################################

        self._core_to_study_id = pd.read_csv(os.path.join(self.raw_dir, dataset.const.CORE_ID_FILENAME))
        self._invalid_study_ids = pd.read_csv(
            os.path.join(self.raw_dir, dataset.const.INVALID_STUDY_IDS_FILENAME)).study_id.values
        graph_files = utils.utils.list_files(self.raw_dir, '.bin')

        unique_study_ids = []
        self._data_files = []
        self.file_name = [] # todo: needs to be implemented
        for f in tqdm(graph_files, disable=not self.show_bar):
            data, is_valid = self._read_record(f)
            if not is_valid:
                continue
            if self._distance_feature:
                data = (utils.graph.concat_edge_distance_features(data[0]),) + data[1:]
            *_, study_id, slide_type = data
            if study_id not in unique_study_ids:
                unique_study_ids.append(study_id)
                self._data_files.append({s: [] for s in self._slide_types})
                self.file_name.append({s: [] for s in self._slide_types})
            study_id_index = unique_study_ids.index(study_id)
            self._data_files[study_id_index][slide_type].append(data[:len(data) - 1])
            self.file_name[study_id_index][slide_type].append(os.path.split(f)[1])
        # remove patients with empty modalities
        invalid_index = [i for i, patient in enumerate(self._data_files) for _, s_v in patient.items() if len(s_v) == 0]
        print('removed study ids: ', [unique_study_ids[i] for i in invalid_index])
        print('removed core ids: ',
              [self._core_to_study_id.loc[self._core_to_study_id.study_id == unique_study_ids[i]].core_id.values[0] for
               i in invalid_index])
        self._data_files = [d for i, d in enumerate(self._data_files) if i not in invalid_index]
        self.file_name = [d for i, d in enumerate(self.file_name) if i not in invalid_index]
        self._study_ids = [d[self._slide_types[0]][0][-1] for d in self._data_files]
        self._censor_status = [d[self._slide_types[0]][0][1] for d in self._data_files]

    @staticmethod
    def _convert_name_to_type(name):
        name = os.path.split(os.path.splitext(name)[0])[1]
        name = name.split('_')[3].split('-')[0]
        return name.lower()

    def _read_record(self, record_path: str):
        slide_type = meta_data.filename_to_stain_name(record_path, self._dataset_name).lower()
        if slide_type not in self._slide_types:
            return None, False

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
        return (graph, censored, partial, time, interval, study_id, slide_type), True

    # def __len__(self):
    #     return len(self._data_files)

    def __getitem__(self, item):
        # record = self._data_files[item]
        record = self._read_h5(item)
        if self._runtime_transforms:
            record = copy.deepcopy(record)
            for k in record:
                for v in range(len(record[k])):
                    for transform in self._runtime_transforms:
                        record[k][v] = (transform(record[k][v][0]),) + record[k][v][1:]
        return record

    def stats(self, obj):
        rep = f"dataset size: {obj.__len__()}\n"

        df = pd.DataFrame(
            {'censor': obj.censor_status, 'study_id': obj.study_ids})

        rep += f"patient count: {len(df.study_id.unique())}\n"
        rep += f"censor count: {len(df.loc[df.censor == True].study_id.unique())}\n"

        rep += f"study ids: {df.study_id.values}\n"

        return rep

    def __repr__(self):
        return self.stats(self)


class MultiModalGraphSubDataset(dgl.data.utils.Subset):

    def __init__(self, dt: MultiModalGraphDataset, ids: typing.List[int],
                 include: bool = True, censor_portion: float = None) -> dgl.data.utils.Subset:
        """
        Selects a subset of the Graph Dataset based on study IDs
        Args:
            dt (MultiModalGraphDataset): input dataset
            ids (List[int]): study ids to select
            include (bool): whether "ids" are included or want to be excluded
        Return:
            (dgl.data.utils.Subset) a dataset with the selected study IDs
        """

        self._transform = None
        self.censor_portion = censor_portion
        dataset_idx = [i for i, d in enumerate(dt.study_ids) if d in ids]
        if not include:
            dataset_idx = [i for i in range(len(dt)) if i not in dataset_idx]
        status = [d for i, d in enumerate(dt.censor_status) if i in dataset_idx]
        self.indexes = {
            'censor': [i for i, c in enumerate(status) if c],
            'event': [i for i, c in enumerate(status) if not c]
        }
        super(MultiModalGraphSubDataset, self).__init__(dt, dataset_idx)

    @property
    def study_ids(self):
        return [s for i, s in enumerate(self.dataset.study_ids) if i in self.indices]

    @property
    def censor_status(self):
        return [s for i, s in enumerate(self.dataset.censor_status) if i in self.indices]

    def set_transform(self, transform):
        if not isinstance(transform, list):
            transform = [transform]
        self._transform = transform

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
        if self._transform:
            data = copy.deepcopy(data)
            for k in data:
                for v in range(len(data[k])):
                    for transform in self._transform:
                        data[k][v] = (transform(data[k][v][0]),) + data[k][v][1:]
        return data

    def __repr__(self):
        # make sensor portion null to go over indexes sequentially
        censor_portion = self.censor_portion
        self.censor_portion = None
        rep = self.dataset.stats(self) + f"censor portion: {self.censor_portion}\n"
        self.censor_portion = censor_portion
        return rep


def split_dataset(dg: MultiModalGraphDataset, split_study_ids: typing.List[int]) -> (GraphDataset, GraphDataset):
    """
    Splits the dataset into two dataset including and excluding the study ID lists
    Args:
        dg (MultiModalGraphDataset): graph dataset to break
        split_study_ids (List[int]): study ID list
    Return:
        (MultiModalGraphDataset, MultiModalGraphDataset): include and exclude datasets
    """
    include_set = MultiModalGraphSubDataset(dg, split_study_ids, include=True)
    exclude_set = MultiModalGraphSubDataset(dg, split_study_ids, include=False)
    return include_set, exclude_set


def k_fold_split_dataset(dg: MultiModalGraphDataset, k: int, censor_portion: float = None, heldout=False) -> typing.List[typing.Tuple]:
    """
    Splits dataset into K-folds based on study ids
    Args:
        dg (GraphDataset): graph dataset
        k (int): number of folds
        censor_portion (float): portion of censor cases in a batch
    Return:
        List[(GraphDataset, GraphDataset)]: list of train/test datasets
    """
    study_ids = dg.study_ids if not heldout else folding.get_heldout_training_study_ids(dg.dataset_name)
    study_ids = np.array(dg.study_ids)
    graphs = []
    for train_study_ids, test_study_ids in folding.get_folded_study_ids(study_ids, k, dg.dataset_name):
        train_graph = MultiModalGraphSubDataset(dg, train_study_ids, censor_portion=censor_portion)
        test_graph = MultiModalGraphSubDataset(dg, test_study_ids) if test_study_ids is not None else None
        graphs.append((train_graph, test_graph))
    return graphs


def get_heldout_folds(dg: MultiModalGraphDataset):
    return {k: MultiModalGraphSubDataset(dg, v) for k, v in folding.get_heldout_study_ids(dg.dataset_name).items()}