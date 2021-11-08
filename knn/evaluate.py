# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import inspect
import math
import os
import re
import shutil
import fitlog
import warnings
import random
from tqdm import tqdm
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, NamedTuple
# Integrations must be imported before ML frameworks:
from transformers.integrations import (  # isort: split
    default_hp_search_backend,
    hp_params,
    is_azureml_available,
    is_comet_available,
    is_fairscale_available,
    is_mlflow_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)

import numpy as np
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import WEIGHTS_NAME, is_datasets_available, is_in_notebook, is_torch_tpu_available
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)

from transformers.trainer_pt_utils import (
    DistributedTensorGatherer,
    SequentialDistributedSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    get_tpu_sampler,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    HPSearchBackend,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
)

import pandas as pd
from utils import *
from training_args import TrainingArguments
from transformers.utils import logging

# Evaluation
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

_is_native_amp_available = False

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available

    if is_apex_available():
        pass
else:
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)

if is_mlflow_available():
    from transformers.integrations import MLflowCallback

    DEFAULT_CALLBACKS.append(MLflowCallback)

if is_optuna_available():
    import optuna

if is_ray_available():
    pass

if is_azureml_available():
    from transformers.integrations import AzureMLCallback

    DEFAULT_CALLBACKS.append(AzureMLCallback)

if is_fairscale_available():
    pass

logger = logging.get_logger(__name__)


filter_words = ['a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'ain', 'all', 'almost',
                'alone', 'along', 'already', 'also', 'although', 'am', 'among', 'amongst', 'an', 'and', 'another',
                'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere', 'are', 'aren', "aren't", 'around', 'as',
                'at', 'back', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides',
                'between', 'beyond', 'both', 'but', 'by', 'can', 'cannot', 'could', 'couldn', "couldn't", 'd', 'didn',
                "didn't", 'doesn', "doesn't", 'don', "don't", 'down', 'due', 'during', 'either', 'else', 'elsewhere',
                'empty', 'enough', 'even', 'ever', 'everyone', 'everything', 'everywhere', 'except', 'first', 'for',
                'former', 'formerly', 'from', 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'he', 'hence',
                'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him', 'himself', 'his',
                'how', 'however', 'hundred', 'i', 'if', 'in', 'indeed', 'into', 'is', 'isn', "isn't", 'it', "it's",
                'its', 'itself', 'just', 'latter', 'latterly', 'least', 'll', 'may', 'me', 'meanwhile', 'mightn',
                "mightn't", 'mine', 'more', 'moreover', 'most', 'mostly', 'must', 'mustn', "mustn't", 'my', 'myself',
                'namely', 'needn', "needn't", 'neither', 'never', 'nevertheless', 'next', 'no', 'nobody', 'none',
                'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'o', 'of', 'off', 'on', 'once', 'one', 'only',
                'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'per',
                'please', 's', 'same', 'shan', "shan't", 'she', "she's", "should've", 'shouldn', "shouldn't", 'somehow',
                'something', 'sometime', 'somewhere', 'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs',
                'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein',
                'thereupon', 'these', 'they', 'this', 'those', 'through', 'throughout', 'thru', 'thus', 'to', 'too',
                'toward', 'towards', 'under', 'unless', 'until', 'up', 'upon', 'used', 've', 'was', 'wasn', "wasn't",
                'we', 'were', 'weren', "weren't", 'what', 'whatever', 'when', 'whence', 'whenever', 'where',
                'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 'wherever', 'whether', 'which', 'while',
                'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'with', 'within', 'without', 'won',
                "won't", 'would', 'wouldn', "wouldn't", 'y', 'yet', 'you', "you'd", "you'll", "you're", "you've",
                'your', 'yours', 'yourself', 'yourselves']
filter_words = set(filter_words)


class MyEvalPrediction(NamedTuple):

    prediction_by_knn: Union[np.ndarray, Tuple[np.ndarray]]
    prediction_by_cls: Union[np.ndarray, Tuple[np.ndarray]]
    prediction_combine: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


def l2norm(x: torch.Tensor):
    norm = torch.pow(x, 2).sum(dim=-1, keepdim=True).sqrt()
    x = torch.div(x, norm)
    return x

class Evaluation:
    """
    This Train is simple version for Origin Trainer.
    This is to exce the origin contrastive learning.
    """
    def __init__(self,
                 model: Union[PreTrainedModel, torch.nn.Module] = None,
                 args: TrainingArguments = None,
                 data_collator: Optional[DataCollator] = None,
                 train_dataset: Optional[Dataset] = None,
                 train_oos_dataset: Optional[Dataset] = None,
                 eval_dataset: Optional[Dataset] = None,
                 test_dataset: Optional[Dataset] = None,
                 tokenizer: Optional["PreTrainedTokenizerBase"] = None,
                 number_labels: Optional[int] = None,
                 eval_oos_dataset: Optional[Dataset] = None,
                 model_init: Callable[[], PreTrainedModel] = None,
                 compute_metrics: Optional[Callable[[MyEvalPrediction], Dict]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
    ):
        if args is None:
            logger.info("No 'TrainingArgumenets' passed, using the current path as 'output_dir'" )
            args = TrainingArguments("tmp_trainer")
        self.args = args
        set_seed(self.args.seed)

        self.number_labels = number_labels
        self.OOS = False
        self.model = model
        default_collator = default_data_collator if tokenizer is None else DataCollatorWithPadding(tokenizer)
        filepath = os.path.join(args.output_dir, 'model_best.pkl')
        self.data_collator = data_collator if data_collator is not None else default_collator
        self.train_dataset = train_dataset
        self.train_oos_dataset = train_oos_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.eval_oos_dataset = eval_oos_dataset
        self.tokenizer = tokenizer
        #self.label_names = default_label_names if self.args.label_names is None else self.args.label_names

    def _get_train_sampler(self) -> Optional[torch.utils.data.sampler.Sampler]:
        if isinstance(self.train_dataset, torch.utils.data.IterableDataset) or not isinstance(
                self.train_dataset, collections.abc.Sized
        ):
            return None
        elif is_torch_tpu_available():
            return get_tpu_sampler(self.train_dataset)
        else:
            return (
                RandomSampler(self.train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(self.train_dataset)
            )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.sampler.Sampler]:
        if is_torch_tpu_available():
            return SequentialDistributedSampler(eval_dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal())
        elif self.args.local_rank != -1:
            return SequentialDistributedSampler(eval_dataset)
        else:
            return SequentialSampler(eval_dataset)

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                If provided, will override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`, columns not
                accepted by the ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        elif is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_dataset, description="evaluation")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_oos_dataloader(self, eval_oos_dataset: Optional[Dataset] = None) -> DataLoader:
        if eval_oos_dataset is None and self.eval_oos_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        elif eval_oos_dataset is not None and not isinstance(eval_oos_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        elif is_datasets_available() and isinstance(eval_oos_dataset, datasets.Dataset):
            self._remove_unused_columns(eval_oos_dataset, description="evaluation")
        eval_oos_dataset = eval_oos_dataset if eval_oos_dataset is not None else self.eval_oos_dataset
        eval_sampler = self._get_eval_sampler(eval_oos_dataset)

        return DataLoader(
            eval_oos_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test :class:`~torch.utils.data.DataLoader`.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (:obj:`torch.utils.data.dataset.Dataset`, `optional`):
                The test dataset to use. If it is an :obj:`datasets.Dataset`, columns not accepted by the
                ``model.forward()`` method are automatically removed. It must implement :obj:`__len__`.
        """
        if not isinstance(test_dataset, collections.abc.Sized):
            raise ValueError("test_dataset must implement __len__")
        elif is_datasets_available() and isinstance(test_dataset, datasets.Dataset):
            pass
            #self._remove_unused_columns(test_dataset, description="test")
        test_sampler = self._get_eval_sampler(test_dataset)

        # We use the same batch_size as for eval.
        return DataLoader(
            test_dataset,
            sampler=test_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )
    # liupeiju
    def get_my_train_dataloader(self, train_dataset: Dataset) -> DataLoader:
        train_sampler = self._get_eval_sampler(train_dataset)

        return DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last
        )
    def get_train_oos_dataloader(self, train_oos_dataset: Dataset) -> DataLoader:
        train_oos_sampler = self._get_eval_sampler(train_oos_dataset)

        return DataLoader(
            train_oos_dataset,
            sampler=train_oos_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            if self.sharded_dpp:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=AdamW,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            else:
                self.optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
            )

    def num_examples(self, dataloader: DataLoader) -> int:
        """
        Helper to get number of samples in a :class:`~torch.utils.data.DataLoader` by accessing its dataset.

        Will raise an exception if the underlying dataset dese not implement method :obj:`__len__`
        """
        return len(dataloader.dataset)

    def evaluation(self, model_path: Optional[str] = None, trial: Union["optuna.Trial", Dict[str, Any]] = None):
        ############## eval in validation ################################
        ############## for the theshold #################################
        predict = []
        target = []

        model = self.model
        #model = torch.load(model_path, map_location=self.args.device)
        valid_loader = self.get_eval_dataloader()
        torch.no_grad()
        model.eval()

        # finetune in valid
        for step, inputs in enumerate(valid_loader):
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)

            output = model(inputs, mode='validation')
            predict += output[1]
            target += output[0]

        f1 = metrics.f1_score(target, predict, average='macro')
        print(f"in-domain f1:{f1}")

        ################### predict ##########################################

        #valid_oos_loader = self.get_eval_oos_dataloader(self.eval_oos_dataset)
        valid_oos_loader = self.get_eval_oos_dataloader()
        train_loader = self.get_train_dataloader()
        test_loader = self.get_test_dataloader(self.test_dataset)

        feature_train = None
        feature_valid = None
        feature_valid_ood = None
        feature_test = None

        prob_train = None
        prob_valid = None
        prob_valid_ood = None
        prob_test = None

        with torch.no_grad():
            y_labels_train = None
            for step, inputs in enumerate(train_loader):
                if y_labels_train is None:
                    y_labels_train = inputs['labels']
                else:
                    y_labels_train = torch.cat((y_labels_train, inputs['labels']), dim=0)


                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                output = model(inputs, mode='test')
                if feature_train != None:
                    feature_train = torch.cat((feature_train, output[1]), dim=0)
                    prob_train = torch.cat((prob_train, output[0]), dim=0)
                else:
                    feature_train = output[1]
                    prob_train = output[0]

            for step, inputs in enumerate(valid_loader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                output = model(inputs, mode='test')
                if feature_valid != None:
                    feature_valid = torch.cat((feature_valid, output[1]), dim=0)
                    prob_valid = torch.cat((prob_valid, output[0]), dim=0)
                else:
                    feature_valid = output[1]
                    prob_valid = output[0]

            for step, inputs in enumerate(valid_oos_loader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                output = model(inputs, mode='test')
                if feature_valid_ood != None:
                    feature_valid_ood = torch.cat((feature_valid_ood, output[1]), dim=0)
                    prob_valid_ood = torch.cat((prob_valid_ood, output[0]), dim=0)
                else:
                    feature_valid_ood = output[1]
                    prob_valid_ood = output[0]

            labels_test = None
            for step, inputs in enumerate(test_loader):
                if labels_test is None:
                    labels_test = inputs['labels']
                else:
                    labels_test = torch.cat((labels_test, inputs['labels']), dim=0)

                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)

                output = model(inputs, mode='test')
                if feature_test != None:
                    feature_test = torch.cat((feature_test, output[1]), dim=0)
                    prob_test = torch.cat((prob_test, output[0]), dim=0)
                    #labels_test = torch.cat((labels_test, inputs['labels']), dim=0)
                else:
                    feature_test = output[1]
                    prob_test = output[0]
                    #labels_test = inputs['labels']

        feature_train = feature_train.cpu().detach().numpy()
        feature_valid = feature_valid.cpu().detach().numpy()
        feature_valid_ood = feature_valid_ood.cpu().detach().numpy()
        feature_test = feature_test.cpu().detach().numpy()
        prob_train = prob_train.cpu().detach().numpy()
        prob_valid = prob_valid.cpu().detach().numpy()
        prob_valid_ood = prob_valid_ood.cpu().detach().numpy()
        prob_test = prob_test.cpu().detach().numpy()

        if self.args.mode == 'find_threshold':
            settings = ['gda_lsqr_' + str(10.0 + 1.0 * (i)) for i in range(20)]
        else:
            if isinstance(self.args.setting, str):
                settings = [self.args.setting, 'gda']

        for setting in settings:
            setting_fields = setting.split("_")
            ood_method = setting_fields[0]
            if ood_method == 'lof':
                lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, metric='cosine', novelty=True, n_jobs=-1)
                lof.fit(feature_train)
                l = len(feature_test)
                y_pred_lof = lof.predict(feature_test)

                #replace oos
                predict_label_test = np.argmax(prob_test, axis=1)
                for ind, val in enumerate(y_pred_lof):
                    if val == -1:
                        predict_label_test[ind] = 150 # oos

                classes = [i for i in range(151)]
                cm = confusion_matrix(labels_test, predict_label_test, classes)
                #f, f_seen, f_unseen, p_unseen, r_unseen = get_score(cm)
                print("this is lof")
                f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen = get_score(cm)
                #plot_confusion_matrix(pred_dir, cm, classes, normalize=False, figsize=(9, 6),
                                      #title=method + ' on ' + dataset + ', f1-macro=' + str(f))
                print(cm)
            elif ood_method == "gda":
                solver = setting_fields[1] if len(setting_fields) > 1 else "lsqr"
                threshold = setting_fields[2] if len(setting_fields) > 2 else "auto"
                distance_type = setting_fields[3] if len(setting_fields) > 3 else "mahalanobis"
                assert solver in ("svd", "lsqr")
                assert distance_type in ("mahalanobis", "euclidean")
                l = len(feature_test)
                method = 'GDA (LMCL)'
                gda = LinearDiscriminantAnalysis(solver=solver, shrinkage=None, store_covariance=True)
                #y_label_train = self.train_dataset['label']
                #y_label_train = self.train_dataset['label']
                gda.fit(prob_train, y_labels_train)
                if threshold == "auto":
                    seen_m_dist = confidence(prob_valid, gda.means_, distance_type, gda.covariance_).min(axis=1)
                    unseen_m_dist = confidence(prob_valid_ood, gda.means_, distance_type, gda.covariance_).min(axis=1)
                    #threshold = estimate_best_threshold(seen_m_dist, unseen_m_dist) - 25
                    threshold = estimate_best_threshold(seen_m_dist, unseen_m_dist)

                else:
                    threshold = float(threshold)

                y_pred = pd.Series(gda.predict(prob_test))
                oos_index_test = []
                for index, val in enumerate(labels_test):
                    if val == 150:
                        oos_index_test.append(index)
                oos_val = []
                ind_val = []
                for index, val in y_pred.items():
                    if index in oos_index_test:
                        oos_val.append(val)
                    else:
                        ind_val.append(val)
                #
                plot_oos_ind(ind_val, oos_val)
                gda_result = confidence(prob_test, gda.means_, distance_type, gda.covariance_)
                #test_info = get_test_info(texts=texts[idx_test[0]:idx_test[0]+l],
                #                          label=y_test[:l],
                #                          label_mask=y_test_mask[:l],
                #                          softmax_prob=prob_test,
                #                          softmax_classes=list(le.classes_),
                #                          gda_result=gda_result,
                #                          gda_classes=gda.classes_,
                #                          save_to_file=True,
                #                          output_dir=pred_dir)

                y_pred_score = pd.Series(gda_result.min(axis=1))
                # plot
                in_domain_dis = './result_value/in_domain_value_05'
                out_domain_dis = './result_value/out_domain_value_05'
                with open(out_domain_dis, 'w+') as out_f1, open(in_domain_dis, 'w+') as in_f1:
                    for index, val in enumerate(labels_test):
                        if val == 150:
                            out_f1.write(str(y_pred_score[index]))
                            out_f1.write('\n')
                        else:
                            in_f1.write(str(y_pred_score[index]))
                            in_f1.write('\n')

                y_pred[y_pred_score[y_pred_score > threshold].index] = 150 ##oos
                #y_label_test = self.test_dataset['label']

                classes = [i for i in range(151)]
                cm = confusion_matrix(labels_test, y_pred, classes)
                print("this is gda")
                f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen = get_score(cm)
                print(cm)
            elif ood_method == "msp":
                threshold = setting_fields[1] if len(setting_fields) > 1 else "auto"
                method = 'MSP (LMCL)'
                l = len(feature_test)
                if threshold == "auto":
                    # prob_valid_seen = model.predict(valid_data[0])
                    # valid_unseen_idx = y_valid[~y_valid.isin(y_cols_seen)].index
                    # prob_valid_unseen = model.predict(X_valid[valid_unseen_idx])
                    seen_conf = prob_valid.max(axis=1) * -1.0
                    unseen_conf = prob_valid_ood.max(axis=1) * -1.0
                    threshold = -1.0 * estimate_best_threshold(seen_conf, unseen_conf)
                else:
                    threshold = float(threshold)

                predict_label_test = np.argmax(prob_test, axis=1)
                label_test_score = np.max(prob_test, axis=1)
                for ind, val in enumerate(label_test_score):
                    if val < threshold:
                        predict_label_test[ind] = 150 # oos

                classes = [i for i in range(151)]
                y_label_test = self.test_dataset['label']
                cm = confusion_matrix(y_label_test, predict_label_test, classes)

                f, acc_all, f_seen, acc_in, p_seen, r_seen, f_unseen, acc_ood, p_unseen, r_unseen = get_score(cm)
                print(cm)

    def save_embed(self):
        model = self.model
        model.eval()

        train_loader = self.get_my_train_dataloader(self.train_dataset)
        train_oos_loader = self.get_train_oos_dataloader(self.train_oos_dataset)
        valid_oos_loader = self.get_eval_oos_dataloader()
        valid_loader = self.get_eval_dataloader()
        test_loader = self.get_test_dataloader(self.test_dataset)

        with torch.no_grad():
            #train
            topk = 25
            embed = []
            for step, inputs in enumerate(train_loader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                _, seq_embed = model(inputs, mode='test')
                embed.append(seq_embed)
            for step, inputs in enumerate(train_oos_loader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                _, seq_embed = model(inputs, mode='test')
                embed.append(seq_embed)
            embed = torch.cat(embed, dim=0)
            torch.save(embed, 'pts/knn_train' + str(topk) + '.pt')
            print('knn_train.pt is saved.')
            #valid
            embed = []
            for step, inputs in enumerate(valid_oos_loader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                _, seq_embed = model(inputs, mode='test')
                embed.append(seq_embed)
            for step, inputs in enumerate(valid_loader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)
                _, seq_embed = model(inputs, mode='test')
                embed.append(seq_embed)
            embed = torch.cat(embed, dim=0)
            torch.save(embed, 'pts/knn_val' + str(topk) + '.pt')
            print('knn_valid.pt is saved.')
            # test
            embed = []
            for step, inputs in enumerate(test_loader):
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        inputs[k] = v.to(self.args.device)

                _, seq_embed = model(inputs, mode='test')
                embed.append(seq_embed)
            embed = torch.cat(embed, dim=0)
            torch.save(embed, 'pts/knn_test' + str(topk) + '.pt')
            print('knn_test.pt is saved.')