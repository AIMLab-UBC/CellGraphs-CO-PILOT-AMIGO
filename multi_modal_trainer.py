import argparse
import copy
import os

import dgl.dataloading
import numpy as np
import torch
import torch.nn.functional
from dgl.dataloading import GraphCollator

import dataset.multi_modal_dataset
import models.gnn
import models.loss
import models.multi_modal_combinator
from models.positional_encoding import ConcatPositionalEncoding, RandLAPositionalEncoding
import models.single_modal_combinator
import models.utils
import utils.distributed
import utils.monitoring
import utils.utils

from models.flop_wrapper import FlopCountAnalysis

# IN_FEATURE_SIZE = 128

torch.multiprocessing.set_sharing_strategy('file_system')


class Trainer:

    def __init__(self):
        self._parse_args()
        utils.distributed.init_distributed_mode(self.args)
        self.monitoring = utils.monitoring.Monitoring(vars(self.args))

        utils.utils.set_seeds(self.args.seed)
        torch.use_deterministic_algorithms(True, warn_only=True)  # for reproducebility purposes
        self._model_dict = self._create_model()
        self.args.batch_size = int(self.args.batch_size / utils.distributed.get_world_size())
        print('world size ', utils.distributed.get_world_size())

    def train(self):
        validation_metrics = []
        # generate_transform = lambda g: utils.utils.homo_to_hetero(g, node_type_count=self.args.hetero_type_count)
        runtime_transforms = self._get_runtime_transforms()
        graph_dataset = dataset.multi_modal_dataset.MultiModalGraphDataset(name=self.args.dataset_unique_name,
                                                                           save_dir=self.args.dataset_save_dir,
                                                                           raw_dir=self.args.data_dir,
                                                                           # transforms=generate_transform,
                                                                           runtime_transforms=runtime_transforms if not self.args.train_only_transform else None,
                                                                           cache=self.args.cache,
                                                                           show_bar=True,
                                                                           slide_types=self.args.modalities,
                                                                           subtypes=self.args.subtypes,
                                                                           distance_feature=self.args.distance_feature,
                                                                           bin_count=self.args.bin_count,
                                                                           hetero_type_count=self.args.hetero_type_count,
                                                                           dataset_name=self.args.dataset_name)
        folds = dataset.multi_modal_dataset.k_fold_split_dataset(graph_dataset, self.args.fold,
                                                                 self.args.batch_censor_portion, self.args.heldout)
        start_fold = 0
        if self.args.fold_number is not None:
            folds = [folds[self.args.fold_number]]
            start_fold = self.args.fold_number

        heldout_folds = dataset.multi_modal_dataset.get_heldout_folds(graph_dataset) if self.args.heldout else {}

        def collate(batch):
            real_dict = {}
            for s in self.args.modalities:
                if s not in real_dict:
                    real_dict[s] = []
                for elem in batch:
                    real_dict[s] += elem[s]
            col = GraphCollator().collate
            new_batch = {k: col(v) for k, v in real_dict.items()}
            return new_batch

        heldout_loaders = {k: dgl.dataloading.GraphDataLoader(v, batch_size=self.args.batch_size, drop_last=False, shuffle=True, num_workers=self.args.worker_count, use_ddp=self.args.distributed, collate_fn=collate) for k, v in heldout_folds.items()} if heldout_folds else {}

        for i, (train_dataset, test_dataset) in enumerate(folds, start=start_fold):
            if self.args.train_only_transform:
                train_dataset.set_transform(runtime_transforms)
            train_loader = dgl.dataloading.GraphDataLoader(train_dataset, batch_size=self.args.batch_size,
                                                           drop_last=False, shuffle=True,
                                                           num_workers=self.args.worker_count,
                                                           use_ddp=self.args.distributed,
                                                           collate_fn=collate)
            if test_dataset is None:
                test_dataset = None
            else:
                test_loader = dgl.dataloading.GraphDataLoader(test_dataset, batch_size=self.args.batch_size,
                                                            drop_last=False, shuffle=True,
                                                            num_workers=self.args.worker_count,
                                                            collate_fn=collate)

            validation_metrics.append(self._train_fold(train_loader, test_loader, heldout_loaders, i))
        print('full fold validation metrics', validation_metrics)

    def _get_runtime_transforms(self):
        from data_workflow.histocartography_extension.graph_builder import NORM_CENTROID

        def rotate(g):
            if not self.args.rotation_aug:
                return g

            # rotate with a random angle
            phi = torch.tensor(np.random.uniform(low=0, high=2) * np.pi)
            s = torch.sin(phi)
            c = torch.cos(phi)
            rot = torch.stack([torch.stack([c, -s]),
                            torch.stack([s, c])])

            g.ndata[NORM_CENTROID] = g.ndata[NORM_CENTROID] @ rot.t()
            return g

        runtime_transforms = [lambda g: dgl.DropNode(self.args.node_drop)(g) if self.args.node_drop else g]
        runtime_transforms.append(rotate)
        return runtime_transforms

    def _train_fold(self, train_loader, test_loader, heldout_loaders, fold_number):
        #models_dict = torch.nn.ModuleDict({s: copy.deepcopy(m) for s, m in self._model_dict.items()})
        models_dict = self._model_dict
        single_modal_combinator = self._create_single_modal_combinator()
        multi_modal_normalizer = self._create_norm_layer(self.args.multi_mode_norm, self.args.graph_dim)
        if self.args.mid_relu and self.args.single_modal_combinator_type != 'average':
            multi_modal_normalizer = torch.nn.Sequential(multi_modal_normalizer, torch.nn.ReLU())
        combinator = self._create_combinator(self.args.graph_dim)
        params = list(combinator.parameters()) + list(multi_modal_normalizer.parameters())
        for _, m in models_dict.items():
            params += list(m.parameters())
        for _, m in single_modal_combinator.items():
            params += list(m.parameters())
        params = set(params)

        params_count = sum(p.numel() for p in params if p.requires_grad)
        print(f'parameter count is {params_count}')

        optimizer = self._set_optimizer(params)
        scheduler = self._set_scheduler(optimizer)

        if self.args.temprature:
            self.temp = np.linspace(2, 0.7, self.args.iteration)
        else:
            self.temp = np.linspace(1, 1, self.args.iteration)

        if self.args.cuda:
            # single_modal_combinator = single_modal_combinator.cuda()
            combinator = combinator.cuda()
            multi_modal_normalizer = multi_modal_normalizer.cuda()
            for s, m in models_dict.items():
                models_dict[s] = m.cuda()
            for s, m in single_modal_combinator.items():
                single_modal_combinator[s] = m.cuda()

        if self.args.distributed:
            for s, m in models_dict.items():
                m = utils.distributed.convert_to_sync_batch_norm(m)
                local_rank = int(os.environ["LOCAL_RANK"])
                models_dict[s] = torch.nn.parallel.DistributedDataParallel(m, device_ids=[local_rank],
                                                                           output_device=local_rank,
                                                                           find_unused_parameters=True)

        print(f'============= fold {fold_number} =====================')
        print('train dataset: ', repr(train_loader.dataset))
        if test_loader is not None:
            print('test dataset: ', repr(test_loader.dataset))
        for k, v in heldout_loaders.items():
            print(f'heldout dataset {k}: {repr(v.dataset)}')

        if self.args.distributed:
            train_loader.set_epoch(0)

        best_c_index = 0
        metric_logger = utils.utils.MetricLogger(delimiter="  ")
        iterator = iter(train_loader)
        for i in metric_logger.log_every(range(self.args.iteration), 10):

            torch.cuda.empty_cache()
            if self.args.distributed and i % len(train_loader) == 0:
                train_loader.set_epoch(int(i // len(train_loader)))

            try:
                data = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                data = next(iterator)

            for s, m in models_dict.items():
                models_dict[s] = m.train()

            # gather patient level info
            study_ids = data[self.args.modalities[0]][-1]
            mapper = {s.item(): i for i, s in enumerate(torch.unique(study_ids))}
            mapper_fn = lambda x: torch.Tensor([mapper[m.item()] for m in x]).to(x.device)

            _, censor, partial, survtime, interval, study_ids = data[self.args.modalities[0]]
            study_ids_reindex = mapper_fn(study_ids)
            multi_modal_survtime = models.utils.aggregate_with_index(survtime, study_ids_reindex, len(mapper))
            multi_modal_interval = models.utils.aggregate_with_index(interval, study_ids_reindex, len(mapper)).type(
                torch.LongTensor)
            multi_modal_censor = models.utils.aggregate_with_index(censor, study_ids_reindex, len(mapper)).type(
                torch.BoolTensor)
            multi_modal_partial = models.utils.aggregate_with_index(partial, study_ids_reindex, len(mapper)).type(
                torch.BoolTensor)

            multi_modal_embedding = torch.empty(0)
            if self.args.cuda:
                device = torch.device('cuda')
                multi_modal_survtime = multi_modal_survtime.to(device)
                multi_modal_interval = multi_modal_interval.to(device)
                multi_modal_censor = multi_modal_censor.to(device)
                multi_modal_partial = multi_modal_partial.to(device)
                multi_modal_embedding = multi_modal_embedding.to(device)

            multi_stage_loss = 0
            for s in self.args.modalities:
                model = models_dict[s]
                graph, censor, partial, survtime, interval, study_ids = data[s]

                # convert study ids to sequential numbers
                study_ids = mapper_fn(study_ids)

                if self.args.cuda:
                    device = torch.device('cuda')
                    graph, censor, partial, survtime, interval, study_ids = (graph.to(device), censor.to(device),
                                                                             partial.to(device), survtime.to(device),
                                                                             interval.to(device), study_ids.to(device))

                embedding, single_embedding_pred, _ = model(graph)

                if self.args.multi_stage_loss:
                    multi_stage_loss += models.loss.cox_loss(survtime, censor, partial,
                                                             single_embedding_pred, decoupling=self.args.dcl)
                # aggregate embeddings
                # Note: this aggregator ensures the alignment of patients within different slide types
                output = single_modal_combinator[s](embedding, study_ids)
                multi_modal_embedding = torch.cat((multi_modal_embedding, output.unsqueeze(1)), dim=1)

            multi_modal_embedding = multi_modal_normalizer(multi_modal_embedding)
            pred = combinator(multi_modal_embedding, temp=self.temp[i])
            if self.args.interval_loss == 'chen':
                loss = models.loss.nll_loss(pred, S=None, Y=multi_modal_interval, c=multi_modal_censor)
                pred = -torch.sum(torch.cumprod(1 - pred, dim=1), dim=1)
            elif self.args.interval_loss == 'hierarchical' or self.args.interval_loss == 'double hierarchical':
                loss = models.loss.cox_loss(multi_modal_survtime, multi_modal_censor, multi_modal_partial, pred[0],
                                            decoupling=self.args.dcl)
                loss += torch.nn.functional.cross_entropy(pred[1], multi_modal_interval)
                pred = pred[0]
            else:
                loss = models.loss.cox_loss(multi_modal_survtime, multi_modal_censor, multi_modal_partial, pred,
                                            decoupling=self.args.dcl)

            loss += self.args.multi_stage_loss * multi_stage_loss

            loss = loss / self.args.gradient_accumulation
            loss.backward()
            if i % self.args.gradient_accumulation == 0 or i == self.args.iteration - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if self.args.cuda:
                torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])
            metric_logger.update(weight_decay=optimizer.param_groups[0]['weight_decay'])

            pred = pred.detach().cpu().numpy().reshape(-1)
            multi_modal_censor = multi_modal_censor.detach().cpu().numpy().reshape(-1)
            multi_modal_survtime = multi_modal_survtime.detach().cpu().numpy().reshape(-1)

            self.monitoring.log({
                'train/loss': loss.item(),
                'train/cindex': utils.utils.c_index_score(pred, multi_modal_censor, multi_modal_survtime),
                'train/pvalue': utils.utils.cox_log_rank(pred, multi_modal_censor, multi_modal_survtime),
                'train/accuracy': utils.utils.accuracy_cox(pred, multi_modal_censor),
            }, step=i)

            epoch = i // len(train_loader)
            if self.args.validation_interval != 0 and (
                    (i % self.args.validation_interval == 0) or (i % len(train_loader) == len(train_loader) - 1)):
                with torch.no_grad():
                    validation_metrics = self._validate(models_dict, combinator, multi_modal_normalizer, single_modal_combinator, test_loader)
                    heldout_metrics = {k: self._validate(models_dict, combinator, multi_modal_normalizer, single_modal_combinator, v) for k, v in heldout_loaders.items()}

                if validation_metrics is not None and validation_metrics['cindex'] > best_c_index:
                    self._save(models_dict, combinator, single_modal_combinator, multi_modal_normalizer, optimizer, validation_metrics['test size'], fold_number)

                for metric, value in validation_metrics.items():
                    if metric == 'df':
                        continue
                    prefix = "metric/"
                    if self.args.fold_number is None:
                        prefix += f"{fold_number}/"
                    # iteration based metrics
                    if i % self.args.validation_interval == 0:
                        self.monitoring.log({f"{prefix}{metric}": value, 'iteration': i, 'epoch': epoch}, step=i)
                    # epoch based metrics
                    if i % len(train_loader) == len(train_loader) - 1:
                        self.monitoring.log({f"epoch_{prefix}{metric}": value, 'iteration': i, 'epoch': epoch}, step=i)
                # save predictions for each iteration
                if i % self.args.validation_interval == 0 and validation_metrics is not None:
                    self.monitoring.save_df(validation_metrics['df'], key=f"prediction_iteration_{i}", step=i)
                    for k, v in heldout_metrics.items():
                        self.monitoring.save_df(v['df'], key=f"heldout/{k}_prediction_iteration_{i}", step=i)
                if i % len(train_loader) == len(train_loader) - 1 and validation_metrics is not None:
                    self.monitoring.save_df(validation_metrics['df'], key=f"prediction_{i // len(train_loader)}", step=i)
                    for k, v in heldout_metrics.items():
                        self.monitoring.save_df(v['df'], key=f"heldout/{k}_prediction_{i // len(train_loader)}", step=i)
                if validation_metrics is not None:
                    print('validation metrics: {}'.format({k: v for k, v in validation_metrics.items() if k != 'df'}))

        metric_logger.synchronize_between_processes()
        with torch.no_grad():
            print(f"Averaged stats for fold {fold_number}: {metric_logger}")
            metric = self._validate(models_dict, combinator, multi_modal_normalizer, single_modal_combinator, test_loader)
            csv_name = os.path.join(self.args.save_dir, f"prediction_{fold_number}.csv")
            if metric is not None:
                metric['df'].to_csv(csv_name, index=False)
                self.monitoring.save_df(metric['df'])

            for k, v in heldout_loaders.items():
                m = self._validate(models_dict, combinator, multi_modal_normalizer, single_modal_combinator, v)
                self.monitoring.save_df(m['df'], key=f"heldout/{k}_prediction_last")

            return metric

    def _validate(self, models_dict, combinator, multi_modal_normalizer, single_modal_combinator, test_loader):

        if test_loader is None:
            return None
        for s, m in models_dict.items():
            models_dict[s] = m.eval()

        risk_pred_all, censor_all, partial_all, survtime_all, study_id_all, loss_list = (
            np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), []
        )

        total_flops = 0
        total_patient_count = 0

        for data in test_loader:

            # gather patient level info
            study_ids = data[self.args.modalities[0]][-1]
            mapper = {s.item(): i for i, s in enumerate(torch.unique(study_ids))}
            mapper_fn = lambda x: torch.Tensor([mapper[m.item()] for m in x]).to(x.device)

            _, censor, partial, survtime, interval, study_ids = data[self.args.modalities[0]]
            study_ids_reindex = mapper_fn(study_ids)
            multi_modal_survtime = models.utils.aggregate_with_index(survtime, study_ids_reindex, len(mapper))
            multi_modal_interval = models.utils.aggregate_with_index(interval, study_ids_reindex, len(mapper)).type(
                torch.LongTensor)
            multi_modal_censor = models.utils.aggregate_with_index(censor, study_ids_reindex, len(mapper)).type(
                torch.BoolTensor)
            multi_modal_partial = models.utils.aggregate_with_index(partial, study_ids_reindex, len(mapper)).type(
                torch.BoolTensor)
            multi_modal_study_id = models.utils.aggregate_with_index(study_ids, study_ids_reindex, len(mapper))

            multi_modal_embedding = torch.empty(0)
            if self.args.cuda:
                device = torch.device('cuda')
                multi_modal_survtime = multi_modal_survtime.to(device)
                multi_modal_interval = multi_modal_interval.to(device)
                multi_modal_censor = multi_modal_censor.to(device)
                multi_modal_partial = multi_modal_partial.to(device)
                multi_modal_study_id = multi_modal_study_id.to(device)
                multi_modal_embedding = multi_modal_embedding.to(device)

            for s in self.args.modalities:
                model = models_dict[s]
                graph, censor, partial, survtime, interval, study_ids = data[s]
                study_ids = mapper_fn(study_ids)

                if self.args.cuda:
                    device = torch.device('cuda')
                    graph, censor, partial, survtime, interval = graph.to(device), censor.to(device), partial.to(
                        device), survtime.to(device), interval.to(device)
                embedding, _, _ = model(graph)
#                total_flops += FlopCountAnalysis(model, graph).total()

                # Note: this aggregator ensures the alignment of patients within different slide types
                output = single_modal_combinator[s](embedding, study_ids)
#                total_flops += FlopCountAnalysis(single_modal_combinator[s], (embedding, study_ids)).total()
                multi_modal_embedding = torch.cat((multi_modal_embedding, output.unsqueeze(1)), dim=1)

            multi_modal_embedding = multi_modal_normalizer(multi_modal_embedding)
            pred = combinator(multi_modal_embedding)
#            total_flops += FlopCountAnalysis(combinator, multi_modal_embedding).total()
            if self.args.interval_loss == 'chen':
                loss = models.loss.nll_loss(pred, S=None, Y=multi_modal_interval, c=multi_modal_censor)
                pred = -torch.sum(torch.cumprod(1 - pred, dim=1), dim=1)
            elif self.args.interval_loss == 'hierarchical' or self.args.interval_loss == 'double hierarchical':
                loss = models.loss.cox_loss(multi_modal_survtime, multi_modal_censor, multi_modal_partial, pred[0],
                                            decoupling=self.args.dcl)
                loss += torch.nn.functional.cross_entropy(pred[1], multi_modal_interval)
                pred = pred[0]
            else:
                loss = models.loss.cox_loss(multi_modal_survtime, multi_modal_censor, multi_modal_partial, pred,
                                            decoupling=self.args.dcl)

            loss_list.append(loss.item())

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, multi_modal_censor.detach().cpu().numpy().reshape(-1)))
            partial_all = np.concatenate((partial_all, multi_modal_partial.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, multi_modal_survtime.detach().cpu().numpy().reshape(-1)))
            study_id_all = np.concatenate((study_id_all, multi_modal_study_id.detach().cpu().numpy().reshape(-1)))

            total_patient_count += multi_modal_study_id.size(0)

        risk_pred_all, censor_all, partial_all, survtime_all, df = utils.utils.hazard_average_by_patient(risk_pred_all,
                                                                                                         censor_all,
                                                                                                         partial_all,
                                                                                                         survtime_all,
                                                                                                         study_id_all,
                                                                                                         drop_partial=True)
        cindex_test = utils.utils.c_index_score(risk_pred_all, censor_all, survtime_all)
        pvalue_test = utils.utils.cox_log_rank(risk_pred_all, censor_all, survtime_all)
        surv_acc_test = utils.utils.accuracy_cox(risk_pred_all, censor_all)

        return {'cindex': cindex_test, 'pvalue': pvalue_test, 'survival acc': surv_acc_test,
                'cox loss': sum(loss_list) / len(loss_list), 'test size': len(risk_pred_all), 'df': df,
                'avg flop patient': total_flops / total_patient_count}

    def _save(self, models_dict, combinator, single_modal_combinator, multi_modal_normalizer, optimizer, test_size, fold):
        models_dict = copy.deepcopy(models_dict)
        for s, m in models_dict.items():
            if isinstance(m, torch.nn.parallel.DistributedDataParallel):
                models_dict[s] = m
        torch.save({
            "model_dict": {s: m.state_dict() for s, m in models_dict.items()},
            "combinator": combinator.state_dict(),
            "multi_modal_normalizer": multi_modal_normalizer.state_dict(),
            "single_modal_combinator": single_modal_combinator.state_dict(),
            "optim": optimizer.state_dict(),
            "test size": test_size
        },
            os.path.join(self.args.save_dir, f"best_model_fold{fold}.pt"))

    def _set_optimizer(self, params):
        self.args.lr = self.args.lr * (self.args.batch_size * utils.distributed.get_world_size()) / 256.
        if self.args.optim == 'adam':
            return torch.optim.Adam(params, lr=self.args.lr, betas=(0.9, 0.99), weight_decay=self.args.weight_decay)
        if self.args.optim == 'sgd':
            return torch.optim.SGD(params, lr=self.args.lr, momentum=0.9, weight_decay=self.args.weight_decay)
        else:
            raise RuntimeError(f'invalid optimizer type {self.args.optim}')

    def _set_scheduler(self, optimizer):
        if self.args.scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.args.iteration)
        else:
            raise RuntimeError(f'invalid scheduler type {self.args.scheduler}')

    def _create_model(self):
        if self.args.arch == 'pathomic':
            model_name = models.gnn.PathomicGraphNet if self.args.hetero_type_count is None else models.gnn.PathomicGraphNetHetero

        shared_module_weights = None
        if self.args.weight_sharing:
            shared_module_weights = [
                {
                    'neigh_shared_weight': torch.nn.Linear(self.args.input_dim, 128, bias=False),
                    'self_shared_weight': torch.nn.Linear(self.args.input_dim, 128, bias=False),
                },
                # {
                #     'neigh_shared_weight': torch.nn.Linear(128, 64, bias=False),
                #     'self_shared_weight': torch.nn.Linear(128, 64, bias=False),
                # },
                # {
                #     'neigh_shared_weight': torch.nn.Linear(128, 64, bias=False),
                #     'self_shared_weight': torch.nn.Linear(128, 64, bias=False),
                # },
            ]

        def inline_creator():
            # ntypes = utils.utils.generate_node_types(self.args.hetero_type_count)
            # etypes = utils.utils.generate_edge_types(self.args.hetero_type_count)
            if self.args.arch == 'gnntransformer':
                return models.gnn.GNNTransformer(in_feat=self.args.input_dim, hidden_feat=128, n_head=1, n_layer=3, position_encoding=self.args.position_encoding, dynamic_graph=self.args.dynamic_graph)

            return model_name(features=self.args.input_dim,
                              weight_sharing=self.args.weight_sharing, node_type_count=self.args.hetero_type_count,
                              edge_weighting=self.args.edge_weighting, conv_type=self.args.conv_type,
                              learnable_skip_connection=self.args.learnable_skip_connection,
                              shared_module_weights=shared_module_weights, position_encoding=self.args.position_encoding, preconv=self.args.preconv,
                              dynamic_graph=self.args.dynamic_graph, dynamic_graph_expansion_scale=self.args.dynamic_graph_expansion_scale,
                              expansion_pos_encoding=self.args.expansion_pos_encoding, pair_norm=self.args.pair_norm, pooling_type=self.args.pooling_type,
                              pooling_ratio=self.args.pooling_ratio, layer_sharing=self.args.layer_sharing, gated_attention=self.args.gated_attention,
                              intra_sharing=self.args.intra_sharing, extra_layer=self.args.extra_layer,
                              expansion_dim_factor=self.args.expansion_dim_factor, plot_graph=self.args.plot_graph, similarity_encoding=self.args.similarity_encoding,
                              hierarchical_attention=self.args.hierarchical_attention, n_layers=self.args.n_layers, fully_connected=self.args.fully_connected,
                              nhid=self.args.nhid, grph_dim=self.args.graph_dim, multi_block=self.args.multi_block, single_layer_preconv=self.args.single_layer_preconv)

        shared_model = inline_creator()
        model_dict = torch.nn.ModuleDict({})
        for s in self.args.modalities:
            model_dict[s] = shared_model if self.args.multi_modal_weight_sharing else inline_creator()
        return model_dict

    def _create_single_modal_combinator(self):
        # IN_FEATURE_SIZE = 128

        def _inside_create():
            if self.args.single_modal_combinator_type == 'average':
                return models.single_modal_combinator.AverageCombinator()
            if self.args.single_modal_combinator_type == 'attention':
                return models.single_modal_combinator.AttentionCombinator(dim_features=self.args.graph_dim)
            if self.args.single_modal_combinator_type == 'transformer':
                return models.single_modal_combinator.TransformerCombinator(dim_features=self.args.graph_dim, n_head=1)
            if self.args.single_modal_combinator_type == 'max':
                return models.single_modal_combinator.MaxCombinator()
            if self.args.single_modal_combinator_type == 'linear':
                return models.single_modal_combinator.Linear(dim_features=self.args.graph_dim)
            raise RuntimeError('invalid single modal combinator type ', self.args.single_modal_combinator_type)

        if self.args.single_combinator_weight_sharing:
            shared = _inside_create()
            return torch.nn.ModuleDict({s: shared for s in self.args.modalities})
        else:
            return torch.nn.ModuleDict({s: _inside_create() for s in self.args.modalities})

    def _create_multi_stage_model(self):
        pass

    def _create_combinator(self, in_features):
        if self.args.combinator_type == 'simple':
            return models.multi_modal_combinator.SimpleCombinator(mode_in_features=in_features,
                                                                  mode_count=len(self.args.modalities),
                                                                  weight_sharing=self.args.combinator_weight_sharing,
                                                                  interval_type=self.args.interval_loss,
                                                                  bin_count=self.args.bin_count)
        elif self.args.combinator_type == 'attention':
            return models.multi_modal_combinator.ModalityAttentionCombinator(mode_in_features=in_features,
                                                                             weight_sharing=self.args.combinator_weight_sharing,
                                                                             interval_type=self.args.interval_loss,
                                                                             bin_count=self.args.bin_count)
        elif self.args.combinator_type == 'transformer':
            return models.multi_modal_combinator.TransformerCombinator(mode_in_features=in_features,
                                                                       weight_sharing=self.args.combinator_weight_sharing,
                                                                       cls_token=self.args.transformer_cls_token,
                                                                       pool=self.args.transformer_pool,
                                                                       n_head=self.args.transformer_n_head,
                                                                       interval_type=self.args.interval_loss,
                                                                       bin_count=self.args.bin_count)
        raise RuntimeError('Combinator type was not expected')

    def _create_norm_layer(self, norm_type, n_feat):
        if norm_type is None:
            return torch.nn.Identity()
        if norm_type == 'batch':
            class RotatedBatchNorm1d(torch.nn.BatchNorm1d):
                def __init__(self, *args, **kwargs):
                    super(RotatedBatchNorm1d, self).__init__(*args, **kwargs)

                def forward(self, x):
                    x = torch.permute(x, (0, 2, 1))
                    x = super().forward(x)
                    x = torch.permute(x, (0, 2, 1))
                    return x

            return RotatedBatchNorm1d(n_feat)
        if norm_type == 'instance':
            return torch.nn.InstanceNorm1d(n_feat)
        if norm_type == 'layer':
            return torch.nn.LayerNorm(n_feat)

    def _parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--data_dir', default='./data', type=str, help="dataset path")
        parser.add_argument('--cuda', default=True, type=utils.utils.bool_flag, help='use cuda')
        parser.add_argument('--save_dir', default='./checkpoint', type=str, help='checkpoint path')
        parser.add_argument('--seed', default=0, type=int, help='random seed')
        parser.add_argument('--job_name', default="", type=str, help='job name')
        parser.add_argument('--slurm_job_id', default="", type=str, help='slurm job id')
        parser.add_argument('--optim', default='adam', type=str.lower, help='optimizer name')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--weight_decay', default=0.0004, type=float, help='weight decay')
        parser.add_argument('--scheduler', default='cosine', type=str.lower, help='scheduler type')
        parser.add_argument('--iteration', default=30000, type=int, help='iterations')
        parser.add_argument('--fold', default=5, type=int, help='number of cross-validation folds')
        parser.add_argument('--fold_number', default=utils.utils.nullable_int_flag, type=int,
                            help='fold number to run (None means all the folds)')
        parser.add_argument('--batch_size', default=512, type=int, help='batch size')
        parser.add_argument('--validation_interval', default=1000, type=int, help='validation interation')
        parser.add_argument('--worker_count', default=0, type=int, help='number of workers')
        parser.add_argument('--hetero_type_count', default=None, type=utils.utils.nullable_int_flag,
                            help='number of node types for heterogeneous graph')
        parser.add_argument('--cache', default=True, type=utils.utils.bool_flag, help='use dataset cache')
        parser.add_argument('--arch', default=None, type=str.lower, help='pathomic | simple')
        parser.add_argument('--input_dim', default=None, type=int, help='input dimension of the graph')
        parser.add_argument('--neptune', default=True, type=utils.utils.bool_flag, help='monitor with neptune')
        parser.add_argument('--wandb', default=True, type=utils.utils.bool_flag, help='monitor with wandb')
        parser.add_argument('--weight_sharing', default=False, type=utils.utils.bool_flag,
                            help='use weight sharing in model')
        parser.add_argument('--multi_modal_weight_sharing', default=False, type=utils.utils.bool_flag,
                            help='use weight sharing across modalities')
        parser.add_argument('--distributed', default=False, type=utils.utils.bool_flag, help='use multi-gpu training')
        parser.add_argument('--tags', nargs='+', default=None, help='monitoring tags')
        parser.add_argument('--dataset_unique_name', type=str, default=None, help='dataset unique name')
        parser.add_argument('--dataset_save_dir', type=str, default=None, help='directory save dir')
        parser.add_argument('--modalities', nargs='+', default=None, help='modality names')
        parser.add_argument('--combinator_weight_sharing', default=False, type=utils.utils.bool_flag,
                            help='use weight sharing for the combinator')
        parser.add_argument('--single_combinator_weight_sharing', default=False, type=utils.utils.bool_flag,
                            help='use weight sharing for the combinator')
        parser.add_argument('--combinator_type', default='simple', type=str.lower,
                            help='combinator type to use for modality combination')
        parser.add_argument('--transformer_cls_token', default=False, type=utils.utils.bool_flag,
                            help='use cls token in transformer')
        parser.add_argument('--transformer_pool', default='mean', type=str.lower,
                            help='transformer pooling type')
        parser.add_argument('--transformer_n_head', default=1, type=int,
                            help='transformer number of attention heads')
        parser.add_argument('--transformer_pos', default=False, type=utils.utils.bool_flag,
                            help='use position encoding in transformer')
        parser.add_argument('--single_modal_combinator_type', default='simple', type=str,
                            help='combinator mode for single modal aggregation')
        parser.add_argument('--subtypes', nargs='+', default=None, help='subtypes')
        parser.add_argument('--edge_weighting', type=utils.utils.nullable_str_flag, default=False, help='use edge information: similarity or distance')
        parser.add_argument('--conv_type', default='sage', type=str.lower, help='type of convolution')
        parser.add_argument('--distance_feature', default=False, type=utils.utils.bool_flag, help='distance features')
        parser.add_argument('--batch_censor_portion', default=None, type=utils.utils.nullable_float_flag,
                            help='portion of censor cases in a batch')
        parser.add_argument('--dcl', default=False, type=utils.utils.bool_flag, help='use decouple contrastive loss')
        parser.add_argument('--multi_stage_loss', default=0, type=float, help='multi stage loss factor')
        parser.add_argument('--auto_gpu', default=False, type=utils.utils.bool_flag, help='select gpus automatically')
        parser.add_argument('--interval_loss', default=None, type=utils.utils.nullable_str_flag,
                            help='interval loss type: chen, hierarchical, confidence, none')
        parser.add_argument('--bin_count', default=4, type=int, help='number of interval bin count')
        parser.add_argument('--learnable_skip_connection', default=False, type=utils.utils.bool_flag,
                            help='use learnable_skip_connection')
        parser.add_argument('--temprature', default=False, type=utils.utils.bool_flag, help='use temprature')
        parser.add_argument('--node_drop', default=None, type=utils.utils.nullable_float_flag,
                            help='probablity of node drop')
        parser.add_argument('--train_only_transform', default=False, type=utils.utils.bool_flag,
                            help='use transform in train only')
        parser.add_argument('--multi_mode_norm', default=None, type=utils.utils.nullable_str_flag,
                            help='normalization for single mode: none, batch, layer, instance')
        parser.add_argument('--mid_relu', default=False, type=utils.utils.bool_flag, help='enable mid relu')
        parser.add_argument('--dataset_name', default="tumorbank", type=str.lower, help='dataset name')
        parser.add_argument('--rotation_aug', default=False, type=utils.utils.bool_flag, help='enable rotation_aug')
        parser.add_argument('--position_encoding', default=None, type=utils.utils.nullable_str_flag, help='position_encoding for point cloud. options: none, concat, randla')
        parser.add_argument('--preconv', default=False, type=utils.utils.bool_flag, help='enable preconv')
        parser.add_argument('--dynamic_graph', default=None, type=utils.utils.nullable_int_flag, help='dynamic graph knn')
        parser.add_argument('--dynamic_graph_expansion_scale', default=0, type=float, help='dynamic graph knn expansion scale')
        parser.add_argument('--expansion_pos_encoding', default=False, type=utils.utils.bool_flag, help='expansion pos encoding')
        parser.add_argument('--pair_norm', default=False, type=utils.utils.bool_flag, help='enable pair normalization')
        parser.add_argument('--pooling_type', default='sag', type=utils.utils.nullable_str_flag, help='pooling type')
        parser.add_argument('--pooling_ratio', default=0.2, type=float, help='pooling ratio')
        parser.add_argument('--layer_sharing', default=False, type=utils.utils.bool_flag, help='layer sharing')
        parser.add_argument('--gated_attention', default='', type=utils.utils.nullable_str_flag, help='enable gated attention: mean, var{n_head}')
        parser.add_argument('--intra_sharing', default=False, type=utils.utils.bool_flag, help='enable intra_sharing')
        parser.add_argument('--heldout', default=False, type=utils.utils.bool_flag, help='enable heldout')
        parser.add_argument('--extra_layer', default=False, type=utils.utils.bool_flag, help='enable extra_layer')
        parser.add_argument('--expansion_dim_factor', default=2, type=float, help='expansion_dim_factor')
        parser.add_argument('--plot_graph', default=False, type=utils.utils.bool_flag, help='plot graph')
        parser.add_argument('--similarity_encoding', default=False, type=utils.utils.bool_flag, help='similarity encoding')
        parser.add_argument('--hierarchical_attention', default=False, type=utils.utils.bool_flag, help='hierarchical_attention')
        parser.add_argument('--n_layers', default=3, type=int, help='num gnn layers')
        parser.add_argument('--fully_connected', default=False, type=utils.utils.bool_flag, help='fully connected')
        parser.add_argument('--nhid', default=256, type=int, help='hidden dim of gnn')
        parser.add_argument('--graph_dim', default=128, type=int, help='output dim of gnn')
        parser.add_argument('--multi_block', default=0, type=int, help='shared block length')
        parser.add_argument('--gradient_accumulation', default=1, type=int, help='gradient accumulation')
        parser.add_argument('--single_layer_preconv', default=False, type=utils.utils.bool_flag, help='single_layer_preconv')
        
        
        

        self.args = parser.parse_args()
        os.makedirs(self.args.save_dir, exist_ok=True)
