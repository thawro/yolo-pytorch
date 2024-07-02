"""DataModule used to load DataLoaders"""

import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.logger.pylogger import log_msg
from src.utils.image import make_grid
from src.utils.types import _split
from src.utils.utils import colorstr


class InfiniteDataLoader(DataLoader):
    """
    Dataloader that reuses workers.

    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        """Dataloader that infinitely recycles workers, inherits from DataLoader."""
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "batch_sampler", _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        """Returns the length of the batch sampler's sampler."""
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        """Creates a sampler that repeats indefinitely."""
        for _ in range(len(self)):
            yield next(self.iterator)

    def reset(self):
        """
        Reset iterator.

        This is useful when we want to modify settings of dataset while training.
        """
        self.iterator = self._get_iterator()


class _RepeatSampler:
    """
    Sampler that repeats forever.

    Args:
        sampler (Dataset.sampler): The sampler to repeat.
    """

    def __init__(self, sampler):
        """Initializes an object that repeats a given sampler indefinitely."""
        self.sampler = sampler

    def __iter__(self):
        """Iterates over the 'sampler' and yields its contents."""
        while True:
            yield from iter(self.sampler)


def seed_worker(worker_id):  # noqa
    """Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class DataModule:
    def __init__(
        self,
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset | None,
        input_size: int,
        batch_size: int = 12,
        num_workers: int = 8,
        pin_memory: bool = True,
        drop_last: bool = True,
        collate_fn=None,
        use_DDP: bool = False,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self._shuffle = None
        self.use_DDP = use_DDP
        self.collate_fn = collate_fn
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds
        self._log_info()
        self.total_batches = {}

    def setup_dataloaders(self):
        self.train_dataloader = self._dataloader("train")
        self.val_dataloader = self._dataloader("val")
        if self.test_ds is not None:
            self.test_dataloader = self._dataloader("test")

    def _log_info(self):
        prefix = colorstr("DataModule") + " stats:"
        datamodule_info = []
        datasets = {"train": self.train_ds, "val": self.val_ds, "test": self.test_ds}
        for split, ds in datasets.items():
            if ds is not None:
                dataset_info = split.rjust(8) + f": {len(ds)} samples ({ds.__class__.__name__})"
            else:
                dataset_info = f"\t{split}: 0 (dataset is None)"
            datamodule_info.append(dataset_info)
        datamodule_info = "\n   ".join(datamodule_info)
        log_msg(f"{prefix}\n{datamodule_info}")

    def _dataloader(self, split: _split):
        shuffle = split == "train"
        if split == "train":
            dataset = self.train_ds
        elif split == "val":
            dataset = self.val_ds
        elif split == "test":
            dataset = self.test_ds
        if dataset is None:
            raise ValueError(f"Dataset for {split} split is None")
        rank = int(os.environ.get("LOCAL_RANK", 0))
        if self.use_DDP:
            log_msg(f"\t{split} dataloader: using DistributedSampler")
            params = dict(
                shuffle=False,
                sampler=DistributedSampler(
                    dataset, shuffle=shuffle, rank=rank, drop_last=self.drop_last
                ),
            )
        else:
            log_msg(f"\t{split} dataloader: Using default Sampler")
            params = dict(shuffle=shuffle)

        # generator = torch.Generator()
        # generator.manual_seed(6148914691236517205 + rank)
        dataloader = InfiniteDataLoader(
            dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            worker_init_fn=seed_worker,
            # generator=generator,
            **params,
        )
        self.total_batches[split] = len(dataloader)
        return dataloader

    def state_dict(self) -> dict:
        return {
            "random_state": random.getstate(),
            "torch_random_state": torch.random.get_rng_state(),
            "torch_cuda_random_state_all": torch.cuda.get_rng_state_all(),
            "torch_cuda_random_state": torch.cuda.get_rng_state(),
            "numpy_random_state": np.random.get_state(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        random.setstate(state_dict["random_state"])
        torch.random.set_rng_state(state_dict["torch_random_state"].cpu())
        torch.cuda.set_rng_state_all(state_dict["torch_cuda_random_state_all"])
        # torch.cuda.set_rng_state(state_dict["torch_cuda_random_state"])
        np.random.set_state(state_dict["numpy_random_state"])
        log_msg("Loaded datamodule state")

    def explore(self):
        import cv2

        dataloader = iter(self.train_dataloader(use_distributed=False))

        def inf_dl_gen():
            while True:
                for batch in dataloader:
                    images, annots, class_idxs = batch
                    images = [self.transform.inverse_preprocessing(img) for img in images]
                    images = make_grid(images, nrows=3)
                    yield images

        gen = inf_dl_gen()
        images = next(gen)
        while True:
            cv2.imshow("Sample", cv2.cvtColor(images, cv2.COLOR_RGB2BGR))
            k = cv2.waitKeyEx(0)
            right_key = 65363
            if k % 256 == 27:  # ESC pressed
                print("Escape hit, closing")
                cv2.destroyAllWindows()
                break
            elif k % 256 == 32 or k == right_key:  # SPACE or right arrow pressed
                print("Space or right arrow hit, exploring next sample")
                images = next(gen)
