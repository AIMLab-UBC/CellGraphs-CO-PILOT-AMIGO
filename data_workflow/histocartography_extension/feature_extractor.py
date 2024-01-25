import copy
from typing import Optional, Callable
from histocartography.preprocessing import feature_extraction

import numpy as np
import torch
from histocartography.preprocessing.feature_extraction import FeatureExtractor, InstanceMapPatchDataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.pathomic_fusion.model import CPC_model


class SequentialFeatureExtractor(FeatureExtractor):

    def __init__(self):
        self._extractors = []

    def add(self, extractors):
        if not isinstance(extractors, list):
            extractors = [extractors]
        self._extractors.extend(extractors)

    def _extract_features(self, input_image: np.ndarray, instance_map: np.ndarray) -> torch.Tensor:
        features = torch.empty(0)
        for extractor in self._extractors:
            features = torch.cat((features, extractor.process(input_image, instance_map)), dim=1)
        return features



class PatchFeatureExtractor(feature_extraction.PatchFeatureExtractor):

    def __init__(self, architecture: str, device: torch.device, patch_size) -> None:
        """
        Create a patch feature extracter of a given architecture and put it on GPU if available.

        Args:
            architecture (str): String of architecture. According to torchvision.models syntax.
            device (torch.device): Torch Device.
        """
        self.device = device

        if architecture.endswith(".jit"):
            self.model = self._get_jit_model(path=architecture)
            self.model.eval()
            self.num_features = self.model(torch.ones(1, 3, patch_size, patch_size).to(device)).size(1)
            return
        super().__init__(architecture, device)

    def _get_jit_model(self, path: str):
        return torch.jit.load(path, map_location=self.device)


class InstanceMapPatchDataset(feature_extraction.InstanceMapPatchDataset):

    def _precompute(self):
        self.threshold = -1
        super()._precompute()

    def _get_patch(self, loc: list, region_id: int = None) -> np.ndarray:
        """
        Extract patch from image.

        Args:
            loc (list): Top-left (x,y) coordinate of a patch.
            region_id (int): Index of the region being processed. Defaults to None. 
        """
        min_x = loc[0]
        min_y = loc[1]
        max_x = min_x + self.patch_size
        max_y = min_y + self.patch_size

        patch = copy.deepcopy(self.image[min_y:max_y, min_x:max_x])
        mask = self.instance_map[min_y:max_y, min_x:max_x]
        patch[(mask != 0) & (mask != region_id)] = self.fill_value

        if self.with_instance_masking:
            instance_mask = ~(self.instance_map[min_y:max_y, min_x:max_x] == region_id)
            patch[instance_mask, :] = self.fill_value

        return patch
    

class DeepFeatureExtractor(feature_extraction.DeepFeatureExtractor):

    def __init__(
        self, 
        device=None,
        **kwargs,
    ) -> None:
        """
        Create a deep feature extractor.

        Args:
            architecture (str): Name of the architecture to use. According to torchvision.models syntax.
            patch_size (int): Desired size of patch.
            resize_size (int): Desired resized size to input the network. If None, no resizing is done and the
                               patches of size patch_size are provided to the network. Defaults to None.
            stride (int): Desired stride for patch extraction. If None, stride is set to patch size. Defaults to None.
            downsample_factor (int): Downsampling factor for image analysis. Defaults to 1.
            normalizer (dict): Dictionary of channel-wise mean and standard deviation for image
                               normalization. If None, using ImageNet normalization factors. Defaults to None.
            batch_size (int): Batch size during processing of patches. Defaults to 32.
            fill_value (int): Constant pixel value for image padding. Defaults to 255.
            num_workers (int): Number of workers in data loader. Defaults to 0.
            verbose (bool): tqdm processing bar. Defaults to False.
        """
        super().__init__(**kwargs)
        if device is not None:
            self.device = device
        self.patch_feature_extractor = PatchFeatureExtractor(
            self.architecture_unprocessed, device=self.device, patch_size=kwargs['patch_size']
        )

    def _extract_features(
        self,
        input_image: np.ndarray,
        instance_map: np.ndarray,
        transform: Optional[Callable] = None
    ) -> torch.Tensor:
        """
        Extract features for a given RGB image and its extracted instance_map.

        Args:
            input_image (np.ndarray): RGB input image.
            instance_map (np.ndarray): Extracted instance_map.
            transform (Callable): Transform to apply. Defaults to None.
        Returns:
            torch.Tensor: Extracted features of shape [nr_instances, nr_features]
        """
        if self.downsample_factor != 1:
            input_image = self._downsample(input_image, self.downsample_factor)
            instance_map = self._downsample(
                instance_map, self.downsample_factor)

        image_dataset = InstanceMapPatchDataset(
            image=input_image,
            instance_map=instance_map,
            resize_size=self.resize_size,
            patch_size=self.patch_size,
            stride=self.stride,
            fill_value=self.fill_value,
            mean=self.normalizer_mean,
            std=self.normalizer_std,
            transform=transform,
            with_instance_masking=self.with_instance_masking,
        )
        image_loader = DataLoader(
            image_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate_patches
        )
        features = torch.empty(
            size=(
                len(image_dataset.properties),
                self.patch_feature_extractor.num_features,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        embeddings = dict()
        for instance_indices, patches in tqdm(
            image_loader, total=len(image_loader), disable=not self.verbose
        ):
            emb = self.patch_feature_extractor(patches)
            for j, key in enumerate(instance_indices):
                if key in embeddings:
                    embeddings[key][0] += emb[j]
                    embeddings[key][1] += 1
                else:
                    embeddings[key] = [emb[j], 1]

        for k, v in embeddings.items():
            features[k, :] = v[0] / v[1]

        return features.cpu().detach()