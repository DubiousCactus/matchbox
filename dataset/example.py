#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Example dataset inheriting from the base ImageDataset class.
This is mostly used to test the framework.
"""

from time import sleep
from typing import Optional, Tuple, Union

import torch
from rich.progress import Progress, TaskID
from torch import Tensor

from dataset.base.image import ImageDataset


class ExampleDataset(ImageDataset):
    IMG_SIZE = (32, 32)

    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        split: str,
        seed: int,
        progress: Progress,
        job_id: TaskID,
        img_dim: Optional[int] = None,
        augment: bool = False,
        normalize: bool = False,
        tiny: bool = False,
        debug: bool = False,
    ) -> None:
        self._img_dim = self.IMG_SIZE[0] if img_dim is None else img_dim
        super().__init__(
            dataset_root,
            dataset_name,
            split,
            seed,
            progress,
            job_id,
            (img_dim, img_dim) if img_dim is not None else None,
            augment=augment,
            normalize=normalize,
            debug=debug,
            tiny=tiny,
        )

    def _load(
        self,
        dataset_root: str,
        tiny: bool,
        split: str,
        seed: int,
        progress: Progress,
        job_id: TaskID,
    ) -> Tuple[Union[dict, list, Tensor], Union[dict, list, Tensor]]:
        progress.update(job_id, total=20)
        for _ in range(20):
            progress.advance(job_id)
            sleep(0.1)
        return torch.rand(10000, self._img_dim, self._img_dim), torch.rand(10000, 8)

    def __getitem__(self, index: int):
        return self._samples[index], self._labels[index]
