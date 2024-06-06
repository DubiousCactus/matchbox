#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Base dataset.
In this file you may implement other base datasets that share the same characteristics and which
need the same data loading + transformation pipeline. The specificities of loading the data or
transforming it may be extended through class inheritance in a specific dataset file.
"""

import abc
import os
import os.path as osp
from typing import Any, Dict, List, Tuple, Union

from hydra.utils import get_original_cwd
from rich.progress import Progress, TaskID
from torch import Tensor
from torch.utils.data import Dataset


class BaseDataset(Dataset[Any], abc.ABC):
    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        augment: bool,
        normalize: bool,
        split: str,
        seed: int,
        progress: Progress,
        job_id: TaskID,
        debug: bool,
        tiny: bool = False,
    ) -> None:
        super().__init__()
        self._samples: Union[Dict[Any, Any], List[Any], Tensor]
        self._labels: Union[Dict[Any, Any], List[Any], Tensor]
        self._progress = progress
        self._samples, self._labels = self._load(
            dataset_root, tiny, split, seed, job_id
        )
        self._augment = augment and split == "train"
        self._normalize = normalize
        self._dataset_name = dataset_name
        self._debug = debug
        self._cache_dir = osp.join(
            get_original_cwd(), "data", f"{dataset_name}_preprocessed"
        )
        os.makedirs(self._cache_dir, exist_ok=True)

    @abc.abstractmethod
    def _load(
        self, dataset_root: str, tiny: bool, split: str, seed: int, job_id: TaskID
    ) -> Tuple[
        Union[Dict[str, Any], List[Any], Tensor],
        Union[Dict[str, Any], List[Any], Tensor],
    ]:
        # Implement this
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._samples)

    def disable_augs(self) -> None:
        self._augment = False
