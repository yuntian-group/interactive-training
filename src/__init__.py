from transformers import Trainer, Seq2SeqTrainer
from src.interactive_training_mixin import InteractiveTrainingMixin


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
