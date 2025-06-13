import time
import uuid
import json
import queue
from typing import Callable
from transformers import TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from trainer.constants import (
    LOAD_CHECKPOINT,
    SAVE_CHECKPOINT,
    STOP_TRAINING,
    UPDATE_OPTIMIZER,
    RESET_LAYER,
    DO_EVALUATE,
    EVENT_MESSAGE_TEMPLATE,
    CHECKPOINT_INFO_UPDATE,
    TRAIN_STATE_UPDATE,
    Cmd,
)


class InteractiveCallbackBase(TrainerCallback):
    def __init__(
        self,
        cmd_queue: queue.Queue,
        event_queue: queue.Queue,
        server_state_update_callback: Callable,
    ):
        """
        Base class for interactive callbacks that handle commands and events.
        :param cmd_queue: Queue for commands from the server.
        :param event_queue: Queue for events to be sent to the server, for broadcast to clients.
        :param server_state_update_callback: Callable that updates the server state with the latest information.
        """
        super().__init__()
        self._cmd_queue = cmd_queue
        self._event_queue = event_queue
        self._server_upate_callback = server_state_update_callback


class InteractiveCallback(InteractiveCallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cur_cmd_list = []

    def _update_optimizer(self, cmd: Cmd):
        """Set the learning rate for the optimizer."""
        update_body_json = cmd.args
        try:
            update_body = json.loads(update_body_json)
            if "lr" in update_body:
                new_lr = update_body["lr"]
                print(f"Updating optimizer learning rate to {new_lr}")
            return True

        except json.JSONDecodeError:
            print(f"Invalid JSON format in command args: {update_body_json}")
            return False

    def on_train_begin(self, args, state, control, **kwargs):
        msg = EVENT_MESSAGE_TEMPLATE.copy()
        msg["command"] = TRAIN_STATE_UPDATE
        cur_model = kwargs.get("model", None)
        model_info = (
            [name for name, module in cur_model.named_modules() if name.strip() != ""]
            if cur_model
            else []
        )

        optimizer_state = {}

        lr_scheduler = kwargs.get("lr_scheduler", None)
        if lr_scheduler:
            optimizer_state["lr"] = lr_scheduler.base_lrs[0]

        all_info = {
            "model_infos": model_info,
            "optimizer_states": optimizer_state,
            "checkpoints": [],
            "command_history": [],
            "uuid": str(uuid.uuid4()),
            "time": time.time(),
        }
        msg["args"] = json.dumps(all_info)

        print("initial info", all_info)

        self._server_upate_callback(msg)

        self._event_queue.put(msg)

    def on_step_end(self, args, state, control, **kwargs):
        while not self._cmd_queue.empty():
            cmd: Cmd = self._cmd_queue.get()
            self._cur_cmd_list.append(cmd)
            if cmd.command == UPDATE_OPTIMIZER:
                if self._update_optimizer(cmd):
                    msg = {
                        "status": "success",
                        "command": UPDATE_OPTIMIZER,
                        "args": cmd.args,
                        "uuid": cmd.uuid,
                        "time": cmd.time,
                    }
                else:
                    msg = {
                        "status": "failed",
                        "command": UPDATE_OPTIMIZER,
                        "args": cmd.args,
                        "uuid": cmd.uuid,
                        "time": cmd.time,
                    }

                self._server_upate_callback(msg)
                self._event_queue.put(msg)

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
        msg_ckpt["args"] = json.dumps({"checkpoint_dir": last_ckpt_dir})
        self._server_upate_callback(msg_ckpt)
        self._event_queue.put(msg_ckpt)


class LoggingCallback(TrainerCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass
