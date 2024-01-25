import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Optional, Any, Tuple, Union

import dgl
import numpy as np
import scipy.stats
import torch
from dgl import load_graphs, save_graphs
from histocartography.preprocessing import graph_builders
from skimage.measure import regionprops, regionprops_table
from sklearn.neighbors import kneighbors_graph

import utils.utils

LABEL = graph_builders.LABEL
CENTROID = graph_builders.CENTROID
NORM_CENTROID = graph_builders.CENTROID + '_norm'
FEATURES = graph_builders.FEATURES
TYPE = 'type'
EDGE_WEIGHT = 'distance'


class BaseGraphBuilder(graph_builders.PipelineStep):
    """
    Base interface class for graph building.
    """

    def __init__(
            self,
            nr_annotation_classes: int = 5,
            annotation_background_class: Optional[int] = None,
            add_loc_feats: bool = False,
            add_type_feats: bool = False,
            num_types: Optional[int] = None,
            **kwargs: Any
    ) -> None:
        """
        Base Graph Builder constructor.
        Args:
            add_type_feats (bool): Flag to include cell type features (ie one-hot type feature)
                                  in node feature representation.
                                  Defaults to False.
            num_types (int): Number of cell types
        """
        self.nr_annotation_classes = nr_annotation_classes
        self.annotation_background_class = annotation_background_class
        self.add_loc_feats = add_loc_feats
        self.add_type_feats = add_type_feats
        self.num_types = num_types
        super().__init__(**kwargs)

    def _process(  # type: ignore[override]
            self,
            instance_map: np.ndarray,
            features: torch.Tensor,
            annotation: Optional[np.ndarray] = None,
            type_map: np.ndarray = None,
            instance_type: dict = None
    ) -> dgl.DGLGraph:
        """Generates a graph from a given instance_map and features
        Args:
            instance_map (np.array): Instance map depicting tissue components
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
            annotation (Union[None, np.array], optional): Optional node level to include.
                                                          Defaults to None.
            type_map (np.array): Instance map depicting cell types.
        Returns:
            dgl.DGLGraph: The constructed graph
        """
        # add nodes
        num_nodes = features.shape[0]
        graph = dgl.DGLGraph()
        graph.add_nodes(num_nodes)

        # add image size as graph data
        image_size = (instance_map.shape[1], instance_map.shape[0])  # (x, y)

        # get instance centroids and types
        centroids, types = self._get_node_centroids_types(instance_map, type_map, instance_type)

        # add node content
        self._set_node_centroids(centroids, image_size, graph)
        if types is not None:
            self._set_node_types(types, graph)
        self._set_node_features(features, image_size, graph)
        if annotation is not None:
            self._set_node_labels(instance_map, annotation, graph)

        # build edges
        self._build_topology(instance_map, centroids, graph)
        return graph

    def _process_and_save(  # type: ignore[override]
            self,
            instance_map: np.ndarray,
            features: torch.Tensor,
            annotation: Optional[np.ndarray] = None,
            type_map: np.ndarray = None,
            output_name: str = None,
    ) -> dgl.DGLGraph:
        """Process and save in provided directory
        Args:
            output_name (str): Name of output file
            instance_map (np.ndarray): Instance map depicting tissue components
                                       (eg nuclei, tissue superpixels)
            features (torch.Tensor): Features of each node. Shape (nr_nodes, nr_features)
            annotation (Optional[np.ndarray], optional): Optional node level to include.
                                                         Defaults to None.
        Returns:
            dgl.DGLGraph: [description]
        """
        assert (
                self.save_path is not None
        ), "Can only save intermediate output if base_path was not None during construction"
        output_path = self.output_dir / f"{output_name}.bin"
        if output_path.exists():
            logging.info(
                f"Output of {output_name} already exists, using it instead of recomputing"
            )
            graphs, _ = load_graphs(str(output_path))
            assert len(graphs) == 1
            graph = graphs[0]
        else:
            graph = self._process(
                instance_map=instance_map,
                features=features,
                annotation=annotation,
                type_map=type_map)
            save_graphs(str(output_path), [graph])
        return graph

    def _get_node_centroids(
            self, instance_map: np.ndarray
    ) -> np.ndarray:
        """Get the centroids of the graphs
        Args:
            instance_map (np.ndarray): Instance map depicting tissue components
        Returns:
            centroids (np.ndarray): Node centroids
        """
        centroids, _ = self._get_node_centroids_types(instance_map)
        return centroids

    def _set_node_centroids(
            self,
            centroids: np.ndarray,
            image_size: tuple,
            graph: dgl.DGLGraph,
    ) -> None:
        """Set the centroids of the graphs
        Args:
            centroids (np.ndarray): Node centroids
            graph (dgl.DGLGraph): Graph to add the centroids to
        """
        centroids = torch.FloatTensor(centroids)
        graph.ndata[CENTROID] = centroids

        # normalized centroids
        normalized_centroids = torch.empty_like(centroids)  # (x, y)
        normalized_centroids[:, 0] = centroids[:, 0] / image_size[0]
        normalized_centroids[:, 1] = centroids[:, 1] / image_size[1]
        graph.ndata[NORM_CENTROID] = normalized_centroids

    def _get_node_centroids_types(
            self,
            instance_map: np.ndarray,
            type_map: np.ndarray = None,
            instance_type: dict = None
    ) -> np.ndarray:
        """Extracts node types from type map
        Args:
            instance_map (np.ndarray): Instance map depicting tissue components
            type_map (np.ndarray): type map of the image
        :return:
            centroids (np.ndarray): Node centroids
            types (np.ndarray): Node types (if type_map not None otherwise None)
        """
        regions = regionprops(instance_map)
        centroids = np.empty((len(regions), 2))
        types = np.empty(len(regions)) if type_map is not None or instance_type is not None else None
        assert instance_type is None or np.max(instance_map) == len(instance_type)
        for i, region in enumerate(regions):
            center_y, center_x = region.centroid  # (y, x)
            center_x = int(round(center_x))
            center_y = int(round(center_y))
            centroids[i, 0] = center_x
            centroids[i, 1] = center_y

        if types is None and instance_type is None:
            return centroids, None

        if instance_type is not None:
            for i, region in enumerate(regions):
                types[i] = instance_type[region.label - 1]
            return centroids, types

        for i, region in enumerate(regions):
            mask = instance_map == region.label
            cell_type = (type_map * mask).flatten()
            non_zero_elements = cell_type[cell_type.nonzero()]
            if len(non_zero_elements) == 0:
                types[i] = 0
            else:
                types[i] = np.median(non_zero_elements) - 1  # deduct one to account for background of 0
        return centroids, types

    def _set_node_types(
            self,
            types: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Sets node types as graph node attributes
        Args:
            types (np.ndarray): types of nodes
            graph (dgl.DGLGraph): graph to add types to
        """
        graph.ndata[TYPE] = torch.LongTensor(types)

    def _set_node_features(
            self,
            features: torch.Tensor,
            image_size: Tuple[int, int],
            graph: dgl.DGLGraph
    ) -> None:
        """Set the provided node features

        Args:
            features (torch.Tensor): Node features
            image_size (Tuple[int,int]): Image dimension (x, y)
            graph (dgl.DGLGraph): Graph to add the features to
        """
        if not torch.is_tensor(features):
            features = torch.FloatTensor(features)

        if self.add_loc_feats:
            assert image_size is not None, "Please provide image size " \
                                           "to add the normalized centroid to the node features."
            # compute normalized centroid features
            centroids = graph.ndata[CENTROID]

            normalized_centroids = torch.empty_like(centroids)  # (x, y)
            normalized_centroids[:, 0] = centroids[:, 0] / image_size[0]
            normalized_centroids[:, 1] = centroids[:, 1] / image_size[1]

            if features.ndim == 3:
                normalized_centroids = normalized_centroids \
                    .unsqueeze(dim=1) \
                    .repeat(1, features.shape[1], 1)

            features = torch.cat((features, normalized_centroids), dim=-1)
        if self.add_type_feats:
            assert TYPE in graph.ndata, 'Please add node type features to the graph.'
            assert self.num_types is not None, 'Please provide total number of cell types.'

            types = graph.ndata[TYPE]
            onehot_type = np.zeros((len(types), self.num_types + 1))
            onehot_type[np.arange(len(types)), types] = 1

            features = torch.cat((features, torch.FloatTensor(onehot_type)), dim=-1)
        graph.ndata[FEATURES] = features

    @abstractmethod
    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Set the node labels of the graphs
        Args:
            instance_map (np.ndarray): Instance map depicting tissue components
            annotation (np.ndarray): Annotations, eg node labels
            graph (dgl.DGLGraph): Graph to add the centroids to
        """

    @abstractmethod
    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Generate the graph topology from the provided instance_map
        Args:
            instance_map (np.array): Instance map depicting tissue components
            centroids (np.array): Node centroids
            graph (dgl.DGLGraph): Graph to add the edges
        """

    def precompute(
            self,
            link_path: Union[None, str, Path] = None,
            precompute_path: Union[None, str, Path] = None,
    ) -> None:
        """Precompute all necessary information
        Args:
            link_path (Union[None, str, Path], optional): Path to link to. Defaults to None.
            precompute_path (Union[None, str, Path], optional): Path to save precomputation outputs. Defaults to None.
        """
        if self.save_path is not None and link_path is not None:
            self._link_to_path(Path(link_path) / "graphs")


class KNNGraphBuilder(BaseGraphBuilder):
    """
    k-Nearest Neighbors Graph class for graph building.
    """

    def __init__(self, k: int = 5, thresh: int = None, **kwargs) -> None:
        """Create a graph builder that uses the (thresholded) kNN algorithm to define the graph topology.

        Args:
            k (int, optional): Number of neighbors. Defaults to 5.
            thresh (int, optional): Maximum allowed distance between 2 nodes. Defaults to None (no thresholding).
        """
        logging.debug("*** kNN Graph Builder ***")
        self.k = k
        self.thresh = thresh
        super().__init__(**kwargs)

    def _set_node_labels(
            self,
            instance_map: np.ndarray,
            annotation: np.ndarray,
            graph: dgl.DGLGraph) -> None:
        """Set the node labels of the graphs using annotation"""
        regions = regionprops(instance_map)
        assert annotation.shape[0] == len(regions), \
            "Number of annotations do not match number of nodes"
        graph.ndata[LABEL] = torch.FloatTensor(annotation.astype(float))

    def _build_topology(
            self,
            instance_map: np.ndarray,
            centroids: np.ndarray,
            graph: dgl.DGLGraph
    ) -> None:
        """Build topology using (thresholded) kNN"""

        # build kNN adjacency
        adj = kneighbors_graph(
            centroids,
            self.k,
            mode="distance",
            include_self=False,
            metric="euclidean",
            n_jobs=-1).toarray()

        # filter edges that are too far (ie larger than thresh)
        if self.thresh is not None:
            adj[adj > self.thresh] = 0

        # make the adjacency matrix symmetric
        adj = utils.utils.make_csr_matrix_symmetric(adj)
        assert utils.utils.is_csr_matrix_symmetric(adj), f'adjacency matrix is not symmetric {adj.toarray()}'

        edge_list = np.nonzero(adj)
        graph.add_edges(list(edge_list[0]), list(edge_list[1]), {EDGE_WEIGHT: torch.FloatTensor(adj[edge_list])})
