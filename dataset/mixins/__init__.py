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
import hashlib
import inspect
import itertools
import os
import os.path as osp
import pickle
import shutil
from multiprocessing.pool import Pool
from os import cpu_count
from typing import Any, Callable, List, Optional, Sequence, Tuple

from hydra.utils import get_original_cwd
from rich.progress import Progress, TaskID
from tqdm import tqdm

from utils.helpers import compressed_read, compressed_write

# TODO: Can we speed all of this up with Cython or Numba?


class DatasetMixinInterface(abc.ABC):
    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        augment: bool,
        normalize: bool,
        split: str,
        seed: int,
        debug: bool,
        tiny: bool,
        progress: Progress,
        job_id: TaskID,
        **kwargs,
    ):
        _ = dataset_root
        _ = dataset_name
        _ = augment
        _ = normalize
        _ = split
        _ = seed
        _ = debug
        _ = tiny
        _ = progress
        _ = job_id
        super().__init__(**kwargs)


class BaseDatasetMixin(DatasetMixinInterface):
    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        augment: bool,
        normalize: bool,
        split: str,
        seed: int,
        debug: bool,
        tiny: bool,
        progress: Progress,
        job_id: TaskID,
        **kwargs,
    ):
        self._samples, self._labels = [], []
        self._augment = augment and split == "train"
        self._normalize = normalize
        self._dataset_name = dataset_name
        self._debug = debug
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
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def disable_augs(self) -> None:
        self._augment = False


# TODO: Automatically "remove" this mixin if debug=True. Idk how, maybe metaclass?
class SafeCacheDatasetMixin(DatasetMixinInterface):
    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        augment: bool,
        normalize: bool,
        split: str,
        seed: int,
        debug: bool,
        tiny: bool,
        progress: Progress,
        job_id: TaskID,
        scd_lazy: bool = True,
        **kwargs,
    ):
        self._cache_dir = osp.join(
            get_original_cwd(),
            "data",
            f"{dataset_name}_preprocessed",
            f"{'tiny_' if tiny else ''}{split}",
        )
        self._split = split
        self._lazy = scd_lazy  # TODO: Implement eager caching (rn the default is lazy)
        # TODO: Compute fingerprint of dataset parameters, data source and *most importantly*
        # code implementation of _load method and every other user function called from it!!!
        # (could we use git for that? Like just querying git diff on the dataset implementation
        # file)
        # If a fingerprint is found in self._cache_dir, compare it to the current fingerprint. If
        # they differ, flush the cache and recompute. If not, load the dataset from cache. If no
        # fingerprint is found, store the current fingerprint.
        argnames = inspect.getfullargspec(SafeCacheDatasetMixin.__init__).args
        frame = inspect.currentframe()
        if frame is None:
            raise RuntimeError("Cannot compute fingerprint without a frame.")
        argvalues = {
            k: v
            for k, v in inspect.getargvalues(frame).locals.items()
            if k in argnames and k not in ["self", "tiny", "scd_lazy"]
        }
        hasher = hashlib.new("md5")
        # TODO: We should also hash the locals of the user's class __init__ method!
        hasher.update(pickle.dumps(argvalues))
        # TODO: Make sure the comments of the user's methods are not included in the fingerprint,
        # and make sure to recursively hash the source code of the user's methods. For the former,
        # we could use the inspect module, for the latter we could use the ast module or a regex
        # with inspect.getcodelines().
        # NOTE: getsource() won't work if I have a decorator that wraps the method. I think it's
        # best to keep this behaviour and not use decorators.
        hasher.update(pickle.dumps(inspect.getsource(self.__class__._load)))  # type: ignore
        hasher.update(pickle.dumps(inspect.getsource(self.__class__._get_raw_elements)))  # type: ignore
        self.fingerprint = hasher.hexdigest()
        mismatch, not_found = False, False
        if osp.isfile(osp.join(self._cache_dir, "fingerprint")):
            with open(osp.join(self._cache_dir, "fingerprint"), "r") as f:
                cached_fingerprint = f.read()
            if cached_fingerprint != self.fingerprint:
                mismatch = True
        else:
            not_found = True

        flush = False
        if mismatch:
            while flush not in ["y", "n"]:
                flush = input("Fingerprint mismatch, flush cache? (y/n) ").lower()
            flush = flush.lower().strip() == "y"
            if not flush:
                print(
                    "[!] Warning: Fingerprint mismatch, but cache will not be flushed."
                )

        if not_found:
            print("No fingerprint found, flushing cache.")
            flush = True

        if flush:
            shutil.rmtree(self._cache_dir, ignore_errors=True)
            os.makedirs(self._cache_dir, exist_ok=True)
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
            **kwargs,
        )
        if flush or not_found:
            with open(osp.join(self._cache_dir, "fingerprint"), "w") as f:
                f.write(self.fingerprint)

    def _get_raw_elements_hook(self, *args, **kwargs) -> Sequence[Any]:
        # TODO: Investigate slowness issues
        class LazyCacheSequence(Sequence):
            def __init__(self, cache_dir: str, seq_len: int, seq_type: str):
                self._cache_paths = []
                self._seq_len = seq_len
                self._seq_type = seq_type

                for i in itertools.count():  # TODO: Could this be slow?
                    cache_path = osp.join(cache_dir, f"{i:04d}.pkl")
                    if not osp.isfile(cache_path):
                        break
                    self._cache_paths.append(cache_path)

                if len(self._cache_paths) != seq_len:
                    raise ValueError(
                        f"Cache info file {osp.join(cache_dir, 'info.txt')} does not match the number of "
                        + f"cache files in {cache_dir}. "
                        + "This may be due to an interrupted dataset computation. "
                        + "Please manually flush the cash to recompute."
                    )

                el_type_str = "unknown"
                try:
                    el_type_str = str(type(compressed_read(self._cache_paths[0])))
                except Exception:
                    pass

                if el_type_str != seq_type:
                    raise ValueError(
                        f"Cache info file {osp.join(cache_dir, 'info.txt')} does not match the type of "
                        + f"cache files in {cache_dir}. "
                        + "This may be due to an interrupted dataset computation. "
                        + "Please manually flush the cash to recompute."
                    )

            def __len__(self):
                return self._seq_len

            def __getitem__(self, idx):
                if idx >= self._seq_len:
                    raise IndexError
                return compressed_read(self._cache_paths[idx])

        # This hooks onto the user's _get_raw_elements method and overrides it if a cache entry is
        # found. If not it just calls the user's _get_raw_elements method.
        # return self._get_raw_elements(*args, **kwargs)
        path = osp.join(self._cache_dir, "raw_elements")
        try:
            info = [None, None]
            if not osp.isfile(osp.join(path, "info.txt")):
                raise FileNotFoundError(
                    f"Cache info file not found at {osp.join(path, 'info.txt')}."
                )
            with open(osp.join(path, "info.txt"), "r") as f:
                info = f.readlines()
            if len(info) != 2 or not info[0].strip().isdigit():
                raise ValueError(
                    f"Invalid cache info file {osp.join(path, 'info.txt')}."
                )
            raw_elements = LazyCacheSequence(
                path, int(info[0].strip()), info[1].strip()
            )
        except FileNotFoundError:
            if not hasattr(self, "_get_raw_elements"):
                raise NotImplementedError(
                    "SafeCacheDatasetMixin._get_raw_elements() is called but the user has not "
                    + f"implemented a _get_raw_elements method in {self.__class__.__name__}."
                )
            # Compute them:
            raw_elements: Sequence[Any] = self._get_raw_elements(*args, **kwargs)  # type: ignore
            type_str = "unknown"
            try:
                type_str = type(raw_elements[0])
            except Exception:
                pass
            # Cache them:
            os.makedirs(path, exist_ok=True)
            with open(osp.join(path, "info.txt"), "w") as f:
                f.writelines([f"{len(raw_elements)}\n", f"{type_str}"])

            print(f"going to cache {len(raw_elements)} elements")
            for i, element in enumerate(raw_elements):
                compressed_write(osp.join(path, f"{i:04d}.pkl"), element)
            print(f"cached {len(raw_elements)} elements")
            # TODO: Rich.log("Raw elements cached here: <>")
        return raw_elements

    def _load_hook(self, *args, **kwargs) -> Tuple[int, Any, Any]:
        # This hooks onto the user's _load method and overrides it if a cache entry is found. If
        # not it just calls the user's _load method.
        idx = args[1]
        cache_path = osp.join(self._cache_dir, f"{idx:04d}.pkl")
        if osp.isfile(cache_path):
            sample, label = None, None
        else:
            if not hasattr(self, "_load"):
                raise NotImplementedError(
                    "SafeCacheDatasetMixin._load() is called but the user has not implemented "
                    + f"a _load method in {self.__class__.__name__}."
                )
            _idx, sample, label = self._load(*args, **kwargs)  # type: ignore
            if _idx != idx:
                raise ValueError(
                    "The _load method returned an index different from the one requested."
                )
        return idx, sample, label

    def _load_sample_label(self, idx: int) -> Tuple[Any, Any]:
        if hasattr(super(), "_load_sample_label"):
            raise Exception(
                "SafeCacheDatasetMixin._load_sample_label() is overriden. "
                + "As best practice, you should inherit from SafeCacheDatasetMixin "
                + f"after {super().__class__.__name__} to avoid unwanted behavior."
            )
        cache_path = osp.join(self._cache_dir, f"{idx:04d}.pkl")
        if not osp.isfile(cache_path):
            raise KeyError(f"Cache file {cache_path} not found, will recompute.")
        return compressed_read(cache_path)

    def _register_sample_label(
        self,
        idx: int,
        sample: Any,
        label: Any,
        memory_samples: List[Any],
        memory_labels: List[Any],
    ):
        if hasattr(super(), "_register_sample_label"):
            raise Exception(
                "SafeCacheDatasetMixin._register_sample_label() is overriden. "
                + "As best practice, you should inherit from SafeCacheDatasetMixin "
                + f"after {super().__class__.__name__} to avoid unwanted behavior."
            )
        cache_path = osp.join(self._cache_dir, f"{idx:04d}.pkl")
        if not osp.isfile(cache_path):
            if sample is None:
                raise ValueError(
                    "The _load_hook method returned sample=None, but no cache entry was found. "
                )
            compressed_write(cache_path, (sample, label))
        memory_samples.insert(idx, cache_path)
        memory_labels.insert(idx, cache_path)


class MultiProcessingDatasetMixin(DatasetMixinInterface, abc.ABC):
    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        augment: bool,
        normalize: bool,
        split: str,
        seed: int,
        debug: bool,
        tiny: bool,
        progress: Progress,
        job_id: TaskID,
        mpd_lazy: bool = True,
        mpd_chunk_size: int = 1,
        mpd_processes: Optional[int] = None,
        **kwargs,
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
            **kwargs,
        )
        self._samples, self._labels = [], []
        cpus = cpu_count()
        processes = (
            1 if debug else (mpd_processes or ((cpus - 1) if cpus is not None else 0))
        )

        with Pool(processes) as pool:
            raw_elements = self._get_raw_elements_hook(
                dataset_root,
                tiny,
                split,
                seed,
            )
            if raw_elements[0] is None or raw_elements[0] is None:
                raise ValueError(
                    "The _get_raw_elements method returned None or a sequence of None. "
                )

            if len(raw_elements) == 0:
                raise ValueError(
                    "The _get_raw_elements method must return a sequence of elements. "
                    + "If the dataset is empty, return an empty list."
                )

            pool_dispatch_method: Callable = pool.imap if mpd_lazy else pool.starmap
            pool_dispatch_func: Callable = (
                self._load_hook_unpack if mpd_lazy else self._load_hook
            )
            for idx, sample, label in tqdm(
                pool_dispatch_method(
                    pool_dispatch_func,
                    zip(
                        raw_elements,
                        # itertools.count(len(raw_elements)),
                        range(len(raw_elements)),
                        itertools.repeat(tiny),
                        itertools.repeat(split),
                        itertools.repeat(seed),
                    ),
                    chunksize=mpd_chunk_size,
                ),
                total=len(raw_elements),
            ):
                self._register_sample_label(idx, sample, label)
        print(f"{'Lazy' if mpd_lazy else 'Eager'} loaded {len(self._samples)} samples.")

    def _get_raw_elements_hook(
        self, dataset_root: str, tiny: bool, split: str, seed: int
    ):
        if hasattr(super(), "_get_raw_elements_hook"):
            return super()._get_raw_elements_hook(dataset_root, tiny, split, seed)  # type: ignore
        else:
            return self._get_raw_elements(dataset_root, tiny, split, seed)

    def _load_hook_unpack(self, args):
        return self._load_hook(*args)

    def _load_hook(self, *args) -> Tuple[int, Any, Any]:
        if hasattr(super(), "_load_hook"):
            return super()._load_hook(*args)  # type: ignore
        else:
            return self._load(*args)  # TODO: Rename to _load_sample?

    def _load_sample_label(self, idx: int) -> Tuple[Any, Any]:
        if hasattr(super(), "_load_sample_label"):
            return super()._load_sample_label(idx)  # type: ignore
        return self._samples[idx], self._labels[idx]

    def _register_sample_label(self, idx: int, sample: Any, label: Any):
        if hasattr(super(), "_register_sample_label"):
            return super()._register_sample_label(  # type: ignore
                idx, sample, label, self._samples, self._labels
            )
        if isinstance(sample, (List, Tuple)) or isinstance(label, (List, Tuple)):
            raise NotImplementedError(
                "_register_sample_label cannot yet handle lists of samples/labels"
            )
        self._samples.insert(idx, sample)
        self._labels.insert(idx, label)

    @abc.abstractmethod
    def _get_raw_elements(
        self, dataset_root: str, tiny: bool, split: str, seed: int
    ) -> Sequence[Any]:
        # Implement this
        raise NotImplementedError

    @abc.abstractmethod
    def _load(
        self, element: Any, idx: int, tiny: bool, split: str, seed: int
    ) -> Tuple[int, Any, Any]:
        # Implement this
        raise NotImplementedError


class BatchedTensorsMultiprocessingDatasetMixin(DatasetMixinInterface, abc.ABC):
    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        augment: bool,
        normalize: bool,
        split: str,
        seed: int,
        debug: bool,
        tiny: bool,
        progress: Progress,
        job_id: TaskID,
        **kwargs,
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
            **kwargs,
        )
        raise NotImplementedError("This is not implemented yet.")
