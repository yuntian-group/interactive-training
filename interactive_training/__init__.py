from transformers import Trainer, Seq2SeqTrainer
from torch.utils.data import Dataset, IterableDataset
from interactive_training.interactive_dataset_mixin import InteractiveDatasetMixin
from interactive_training.interactive_training_mixin import InteractiveTrainingMixin


def make_interactive(base_cls):
    if not issubclass(base_cls, Trainer) and not issubclass(base_cls, Seq2SeqTrainer):
        raise TypeError(
            f"Base class must be a subclass of Trainer, got {base_cls.__name__}"
        )

    class InteractiveTrainer(InteractiveTrainingMixin, base_cls):
        pass

    InteractiveTrainer.__name__ = f"Interactive{base_cls.__name__}"
    InteractiveTrainer.__qualname__ = InteractiveTrainer.__name__

    return InteractiveTrainer


def make_interactive_dataset(base_cls):
    if not issubclass(base_cls, Dataset) and not issubclass(base_cls, IterableDataset):
        raise TypeError(
            f"Base class must be a subclass of Dataset, got {base_cls.__name__}"
        )

    class InteractiveDataset(InteractiveDatasetMixin, base_cls):
        pass

    InteractiveDataset.__name__ = f"Interactive{base_cls.__name__}"
    InteractiveDataset.__qualname__ = InteractiveDataset.__name__
    return InteractiveDataset
