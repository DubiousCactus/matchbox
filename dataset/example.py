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


def test():
    # return
    raise Exception("This is an exception")


def test_target():
    return
    raise Exception("This is an exception")


def test_recursive():
    # raise Exception("This is an exception")
    test_target()


class SingleProcessingExampleDataset(ImageDataset):
    IMG_SIZE = (32, 32)

    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        split: str,
        seed: int,
        progress: Optional[Progress] = None,
        job_id: Optional[TaskID] = None,
        img_dim: Optional[int] = None,
        augment: bool = False,
        normalize: bool = False,
        tiny: bool = False,
        debug: bool = False,
    ) -> None:
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
        # raise NotImplementedError("This is a dummy error")
        self._img_dim = self.IMG_SIZE[0] if img_dim is None else img_dim
        self._samples, self._labels = self._load(
            progress,
            job_id,
        )

    def test(self):
        print(f"self.__getitem__(0): x={self[0][0].shape}, y{self[0][1].shape}")

    def _load(
        self,
        progress: Optional[Progress] = None,
        job_id: Optional[TaskID] = None,
    ) -> Tuple[Union[dict, list, Tensor], Union[dict, list, Tensor]]:
        length = 3 if self._tiny else 20
        if progress is not None:
            assert job_id is not None
            progress.update(job_id, total=length)
        # raise Exception("This is an exception")
        # test()
        test_recursive()
        for _ in range(length):
            if progress is not None:
                assert job_id is not None
                progress.advance(job_id)
            sleep(0.001 if self._tiny else 0.1)
        return torch.rand(10000, self._img_dim, self._img_dim), torch.rand(10000, 8)

    def __getitem__(self, index: int):
        return self._samples[index], self._labels[index]


class MultiProcessingExampleDataset(ImageDataset):  # TODO
    IMG_SIZE = (32, 32)

    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        split: str,
        seed: int,
        progress: Optional[Progress] = None,
        job_id: Optional[TaskID] = None,
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
        # TODO:


class MultiProcessingWithCachingExampleDataset(ImageDataset):  # TODO
    IMG_SIZE = (32, 32)

    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        split: str,
        seed: int,
        progress: Optional[Progress] = None,
        job_id: Optional[TaskID] = None,
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
        # TODO:
