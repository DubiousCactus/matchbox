#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2023 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Base dataset for images.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from rich.progress import Progress, TaskID
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io.image import read_image  # type: ignore
from torchvision.transforms import transforms  # type: ignore

from dataset.mixins import BaseDatasetMixin


class ImageDataset(BaseDatasetMixin, Dataset):
    IMAGE_NET_MEAN: List[float] = []
    IMAGE_NET_STD: List[float] = []
    COCO_MEAN, COCO_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    IMG_SIZE = (32, 32)

    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        split: str,
        seed: int,
        progress: Optional[Progress] = None,
        job_id: Optional[TaskID] = None,
        img_size: Optional[tuple[int, ...]] = None,
        augment: bool = False,
        normalize: bool = False,
        tiny: bool = False,
        debug: bool = False,
    ) -> None:
        super().__init__(
            dataset_root,
            dataset_name,
            augment,
            normalize,
            split,
            seed,
            debug,
            tiny,
            progress,
            job_id,
        )
        self._img_size = self.IMG_SIZE if img_size is None else img_size
        self._transforms: Callable[[Tensor], Tensor] = transforms.Compose(
            [
                transforms.Resize(self._img_size),
            ]
        )
        self._normalization: Callable[[Tensor], Tensor] = transforms.Normalize(
            self.IMAGE_NET_MEAN, self.IMAGE_NET_STD
        )
        if self._augment:
            try:
                import albumentations as A  # type: ignore
            except ImportError:
                raise ImportError(
                    "Please install albumentations to use the augmentation pipeline."
                )
            self._augs: Callable[..., Dict[str, Any]] = A.Compose(
                [
                    A.RandomCropFromBorders(),
                    A.RandomBrightnessContrast(),
                    A.RandomGamma(),
                ]
            )

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """
        This should be common to all image datasets!
        Override if you need something else.
        """
        # ==== Load image and apply transforms ===
        img: Tensor
        img = read_image(self._samples[index])  # type: ignore
        img = self._transforms(img)
        if self._normalize:
            img = self._normalization(img)
        if self._augment:
            img = self._augs(image=img)["image"]
        # ==== Load label and apply transforms ===
        label: Any = self._labels[index]
        return img, label
