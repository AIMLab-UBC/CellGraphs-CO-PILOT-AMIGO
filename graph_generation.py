"""
Extract HACT graphs for all the sample in the BRACS dataset.
"""

import argparse
import multiprocessing
import os
import time
from glob import glob

import numpy as np
import scipy.io
from PIL import Image
from dgl.data.utils import save_graphs
from histocartography.preprocessing import (
    # stain normalizer
    # nuclei detector
    # feature extractor
    # KNNGraphBuilder,  # kNN graph builder
    # tissue detector
    # DeepFeatureExtractor,  # feature extractor
    HandcraftedFeatureExtractor,  # hand-crafted extractor
    # build graph
    # assignment matrix
)
import torch
from tqdm import tqdm

import utils.utils
from data_workflow.histocartography_extension.feature_extractor import SequentialFeatureExtractor, CPCFeatureExtractor, DeepFeatureExtractor
from data_workflow.histocartography_extension.graph_builder import KNNGraphBuilder
from data_workflow.histocartography_extension.nuclei_extractor import NucleiExtractorPrecomputed
from data_workflow.histocartography_extension.nuclei_filter import NucleiFilter

MIN_NR_PIXELS = 50000
MAX_NR_PIXELS = 50000000
STAIN_NORM_TARGET_IMAGE = '../data/target.png'  # define stain normalization target image.


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='', required=True, help='path to the data.')
    parser.add_argument('--instance_mask_path', type=str, default=None, help='path to the instance mask.')
    parser.add_argument('--image_extension', type=str, default='.tif', help='image extension')
    parser.add_argument('--instance_mask_extension', type=str, default='.npz', help='instance mask extension')
    parser.add_argument('--graph_k', type=int, default=5, help='number of neighbors in graph')
    parser.add_argument('--graph_max_distance', type=utils.utils.nullable_int_flag, default=50,
                        help='max pixel distance threshold')
    parser.add_argument('--cell_min_area', type=int, default=None, help='smallest possible cell area (None to disable)')
    parser.add_argument('--cell_max_area', type=int, default=None, help='largest possible cell area (None to disable)')
    parser.add_argument('--ihc_stain_ratio', type=float, default=None,
                        help='ratio of ihc stain (hematoxylin or DAB) in comparison to the whole area of the cell to '
                             'check the validity of nuclei')
    parser.add_argument('--num_cell_types', type=int, default=2, help='number of cell types in the graph')
    parser.add_argument('--save_path', type=str, default='../data/', required=True, help='path to save the graphs.')
    parser.add_argument('--workers', type=int, default=10, help='workers count')
    parser.add_argument('--handcraft_features', type=utils.utils.bool_flag, default=False,
                        help='use handcraft features')
    parser.add_argument('--cpc_path', type=str, default=None, help='path to the cpc pretrained model')
    parser.add_argument('--deep_feature_arch', type=utils.utils.nullable_str_flag_none_lower, default='resnet34',
                        help='architecture for deep features')
    parser.add_argument('--add_location', type=utils.utils.bool_flag, default=False, help='add location info')
    parser.add_argument('--start_pattern', type=str, default="", help='starting pattern of the file')
    parser.add_argument('--num_feature_workers', type=int, default=0, help='num feature workers')
    parser.add_argument('--patch_size', type=int, default=72, help='patch size for the cell')
    parser.add_argument('--norm_mean', metavar='N', type=float, nargs='+', help='norm_mean')
    parser.add_argument('--norm_std', metavar='N', type=float, nargs='+', help='norm_std')
    return parser.parse_args()


class GraphBuilder:

    def __init__(self, args):

        # set the configuration
        self.config = args

        # define CG builders
        self._build_cg_builders()

        # 4. define var to store image IDs that failed (for whatever reason)
        self.image_ids_failing = []

    def _build_cg_builders(self):
        # a define nuclei extractor
        self.nuclei_detector = NucleiExtractorPrecomputed()

        # setup nuclei filter
        self.nuclei_filter = NucleiFilter(smallest_area=self.config.cell_min_area,
                                          largest_area=self.config.cell_max_area,
                                          ihc_stain_ratio=self.config.ihc_stain_ratio)

        # b define feature extractor: Extract patches of 72x72 pixels around each
        # nucleus centroid, then resize to 224 to match ResNet input size.
        self.nuclei_feature_extractor = [SequentialFeatureExtractor() for _ in range(self.config.workers)]
        n_devices = torch.cuda.device_count()
        for i in range(self.config.workers):
            self.create_feature_extractor(self.nuclei_feature_extractor[i], torch.device(f'cuda:{i % n_devices}' if n_devices > 0 else "cpu") )

        # c define k-NN graph builder with k=5 and thresholding edges longer
        # than 50 pixels. Add image size-normalized centroids to the node features.
        # For e.g., resulting node features are 512 features from ResNet34 + 2
        # normalized centroid features.
        self.knn_graph_builder = KNNGraphBuilder(k=self.config.graph_k, thresh=self.config.graph_max_distance,
                                                 add_loc_feats=self.config.add_location,
                                                 add_type_feats=self.config.num_cell_types > 0,
                                                 num_types=self.config.num_cell_types)

    def create_feature_extractor(self, extractor, device):
        if self.config.deep_feature_arch:
            extractor.add(DeepFeatureExtractor(
                architecture=self.config.deep_feature_arch,
                patch_size=self.config.patch_size,
                resize_size=224,
                num_workers=0,
                batch_size=50,
                fill_value=255,
                normalizer={'mean': self.config.norm_mean, 'std': self.config.norm_std},
                device=device
                ))
        if self.config.handcraft_features:
            extractor.add(HandcraftedFeatureExtractor())
        if self.config.cpc_path:
            extractor.add(CPCFeatureExtractor(
                architecture=self.config.cpc_path,
                patch_size=64
                ))

    def _build_cg(self, image, instance_mask=None, type_map=None, instance_type=None, worker_id=0):
        nuclei_map, _ = self.nuclei_detector.process(image, instance_mask=instance_mask)
        nuclei_map = self.nuclei_filter.process(image, nuclei_map)
        features = self.nuclei_feature_extractor[worker_id].process(image, nuclei_map)
        graph = self.knn_graph_builder.process(nuclei_map, features, type_map=type_map, instance_type=instance_type)
        return graph

    @staticmethod
    def _read_mask(file_name, n_cell_type):
        ext = os.path.splitext(os.path.split(file_name)[1])[1]
        if ext == '.mat':
            instance_mask = scipy.io.loadmat(file_name)
            # type_mask = instance_mask['type_mask'].toarray()
            instance_type = instance_mask['inst_type']
            if 'inst_mask' in instance_mask:
                instance_mask = instance_mask['inst_mask'].toarray()
            else:
                instance_mask = instance_mask['inst_map']
            if len(np.unique(instance_mask)) == len(instance_type) + 2:  # because of weird problem in HoverNet
                instance_mask[instance_mask == np.max(instance_mask)] = 0
            return instance_mask, None, instance_type
        instance_mask = np.load(file_name)
        # if it's zipped, extract from dictionary
        if '.npz' in ext:
            instance_mask = instance_mask['arr_0']
        return instance_mask[..., 0], instance_mask[..., 1] if n_cell_type > 0 else None, None

    def worker_process(self, image_file, image_path, instance_mask_path, save_path, worker_id):
        try:
            # a. load image & check if already there
            _, image_name = os.path.split(image_file)
            image = np.array(Image.open(image_file))

            instance_mask = None
            type_mask = None
            if instance_mask_path is not None:
                # replace the parrent dir
                instance_mask = os.path.join(instance_mask_path, os.path.relpath(image_file, image_path))
                # replace the extension
                instance_mask = instance_mask.replace(self.config.image_extension, self.config.instance_mask_extension)
                # read the masks
                instance_mask, type_mask, instance_type = self._read_mask(instance_mask, self.config.num_cell_types)

            # extract meta data
            nr_pixels = image.shape[0] * image.shape[1]
            # patient_id = meta_data.voa_008_filename_to_patient_id(image_name)

            cg_out = os.path.join(save_path, image_name.replace(self.config.image_extension, '.bin'))

            # if file was not already created + not too big + not too small, then process
            if not self._exists(cg_out) and self._valid_image(nr_pixels):
                    cell_graph = self._build_cg(image, instance_mask=instance_mask, type_map=type_mask,
                                                instance_type=instance_type, worker_id=worker_id)
                    save_graphs(
                        filename=cg_out,
                        g_list=[cell_graph]
                        # labels={"patient_id": torch.tensor([patient_id])}
                    )
            else:
                print('Image:', image_file, ' was already processed or is too large/small.')
        except Exception as e:
            print(f'exception raised in processing {image_file}: {e}')

    def start_threads(self, image_path, instance_mask_path, save_path, queue, worker_id):
        while True:
            data = queue.get()
            if not isinstance(data, str):
                time.sleep(1)
                continue
            if data == "done":
                return
            self.worker_process(data, image_path, instance_mask_path, save_path, worker_id)

    def process(self, image_path, instance_mask_path, save_path, pattern):
        # 1. get image path
        subdirs = os.listdir()
        image_fnames = []
        for subdir in (subdirs + ['']):  # look for all the subdirs AND the image path
            image_fnames += glob(os.path.join(image_path, subdir, f'{pattern}*{self.config.image_extension}'))

        print('*** Start analysing {} images ***'.format(len(image_fnames)))

        if args.workers <= 1:
            for img_file in tqdm(image_fnames, total=len(image_fnames)):
                self.worker_process(img_file, image_path, instance_mask_path, save_path, 0)
            return

        workers = []
        multiprocessing.set_start_method('spawn')
        queue = multiprocessing.Queue()
        for i in range(args.workers):
            p = multiprocessing.Process(target=self.start_threads,
                                        args=((image_path, instance_mask_path, save_path, queue, i)))
            p.daemon = True
            p.start()
            workers.append(p)
        for img in image_fnames:
            queue.put(img)
        for i in range(len(workers)):
            queue.put('done')
        for i, worker in enumerate(workers):
            worker.join()

        #        with multiprocessing.Pool(args.workers) as pool:
        #            pool.map(functools.partial(self.worker_process, image_path=image_path,
        #                                       instance_mask_path=instance_mask_path, save_path=save_path), image_fnames)

        print('Out of {} images, {} successful HACT graph generations.'.format(
            len(image_fnames), len(image_fnames) - len(self.image_ids_failing)))
        print('Failing IDs are:', self.image_ids_failing)

    def _valid_image(self, nr_pixels):
        if MIN_NR_PIXELS < nr_pixels < MAX_NR_PIXELS:
            return True
        return False

    def _exists(self, cg_out):
        if os.path.isfile(cg_out):
            return True
        return False


if __name__ == "__main__":

    # 1. handle i/o
    args = parse_arguments()
    print('args', args)
    if not os.path.isdir(args.image_path) or not os.listdir(args.image_path):
        raise ValueError("Data directory is either empty or does not exist.")

    if args.instance_mask_path is not None and \
            not os.path.isdir(args.instance_mask_path) or not os.listdir(args.instance_mask_path):
        raise ValueError("Instance mask directory is either empty or does not exist.")

    # 2. generate HACT graphs one-by-one, will automatically
    # run on GPU if available.
    hact_builder = GraphBuilder(args)
    hact_builder.process(args.image_path, args.instance_mask_path, args.save_path, args.start_pattern)

    with open(os.path.join(args.save_path, 'config.txt'), 'w') as f:
        f.write(vars(args))
