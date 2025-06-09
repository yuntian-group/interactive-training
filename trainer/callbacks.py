import queue
from typing import Union
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from trainer.constants import (
    LOAD_CHECKPOINT,
    SAVE_CHECKPOINT,
    STOP_TRAINING,
    SET_LR,
    RESET_LAYER,
    DO_EVALUATE,
    EVENT_MESSAGE_TEMPLATE,
    CHECKPOINT_INFO_UPDATE,
    TRAIN_INFO_UPDATE,
    Cmd,
)


class InteractiveCallbackBase(TrainerCallback):
    def __init__(
        self,
        cmd_queue: queue.Queue,
        event_queue: queue.Queue,
        model=None,
        optimizer=None,
        lr_scheduler=None,
        train_dataloader=None,
        eval_dataloader=None,
    ):
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader
        self._cmd_queue = cmd_queue
        self._event_queue = event_queue


class InteractiveCallback(InteractiveCallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cur_cmd_list = []

    def _set_lr(self, cmd: Cmd):
        """Set the learning rate for the optimizer."""

        lr = cmd.args
        if isinstance(lr, str):
            lr = float(lr)
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Learning rate set to {lr}")

    def on_step_end(self, args, state, control, **kwargs):
        while not self._cmd_queue.empty():
            cmd: Cmd = self._cmd_queue.get()
            self._cur_cmd_list.append(cmd)
            if cmd.command == SET_LR:
                self._set_lr(cmd)

            elif cmd.command == STOP_TRAINING or cmd.command == LOAD_CHECKPOINT:
                control.should_training_stop = True
                print("Training stopped by command.")

            elif cmd.command == RESET_LAYER:
                print("Resetting layer weights is not implemented yet.")

            elif cmd.command == DO_EVALUATE:
                control.should_evaluate = True


class CheckpointCallback(InteractiveCallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._is_cmd_save = False

    def on_step_begin(self, args, state, control, **kwargs):
        while not self._cmd_queue.empty():
            cmd: Cmd = self._cmd_queue.get()
            if cmd.command == SAVE_CHECKPOINT:
                control.should_save = True
                self._is_cmd_save = True
                print("Training stopped by command.")
            else:
                print(f"Unknown command for stop callback: {cmd.command}")

    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        last_ckpt_dir = get_last_checkpoint(args.output_dir)
        if self._is_cmd_save:
            self._is_cmd_save = False

            control.should_save = False  # Reset to avoid multiple saves
            msg = EVENT_MESSAGE_TEMPLATE.copy()
            msg["command"] = SAVE_CHECKPOINT
            self._event_queue.put(msg)

        msg_ckpt = EVENT_MESSAGE_TEMPLATE.copy()
        msg_ckpt["command"] = CHECKPOINT_INFO_UPDATE
        msg_ckpt["metadata"] = {"checkpoint_dir": last_ckpt_dir}
        self._event_queue.put(msg_ckpt)


class LoggingCallback(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass


class InfoCollectCallback(InteractiveCallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_train_begin(self, args, state, control, **kwargs):
        msg = EVENT_MESSAGE_TEMPLATE.copy()
        msg["command"] = TRAIN_INFO_UPDATE
        msg["metadata"] = {
            "model_info": [
                name
                for name, module in self._model.named_modules()
                if name.strip() != ""
            ],
        }

        print(f"Model info: {str(msg)}")

        self._event_queue.put(msg)
