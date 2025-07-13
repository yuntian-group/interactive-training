import os
import time
import uuid
import json
import queue
import torch
from collections import deque
from typing import Callable, List
from transformers import TrainerCallback
from transformers.utils import logging
from transformers.trainer_utils import get_last_checkpoint
from interactive_training.constants import (
    LOAD_CHECKPOINT,
    SAVE_CHECKPOINT,
    STOP_TRAINING,
    UPDATE_OPTIMIZER,
    RESET_LAYER,
    DO_EVALUATE,
    EVENT_MESSAGE_TEMPLATE,
    CHECKPOINT_INFO_UPDATE,
    TRAIN_STATE_UPDATE,
    CMD_SUCCESS,
    CMD_RUNNING,
    CMD_FAILED,
    PAUSE_TRAINING,
    RESUME_TRAINING,
    UPDATE_DATASET_RUNTIME_HYPERPARAMETERS,
    INIITAL_DATASET_INFORMATION,
    Cmd,
)
from interactive_training.interactive_dataset_mixin import InteractiveDatasetMixin

logger = logging.get_logger(__name__)


class InteractiveCallbackBase(TrainerCallback):
    def __init__(
        self,
        cmd_queue: queue.Queue = None,
        event_queue: queue.Queue = None,
        server_state_update_callback: Callable = None,
        current_branch_id_callback: Callable = None,
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
        self._server_update_callback = server_state_update_callback
        self._current_branch_id_callback = current_branch_id_callback


class InteractiveCallback(InteractiveCallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._cur_cmd_list = []
        self._cmd_evaluate = False

    def _parse_model_tree(self, module_names: List[str], model_type_dict: dict):
        """Parse a list of module names into a tree structure."""
        tree = {}
        parent = {}
        module_names_split = list(
            sorted([name.split(".") for name in module_names], key=lambda x: len(x))
        )

        # build adjacency list
        for name_parts in module_names_split:
            cur_name = ".".join(name_parts)
            if len(name_parts) == 0:
                continue

            tree[cur_name] = []
            if len(name_parts) > 1:
                parent_name = ".".join(name_parts[:-1])
                parent_children_list = tree[parent_name]
                parent_children_list.append(cur_name)
                parent[cur_name] = parent_name

        roots_list = []
        for name in module_names:
            if name not in parent:
                roots_list.append(name)

        root_tree_node = {
            "name": None,
            "children": [],
            "module_type": model_type_dict.get(name, "Unknown"),
        }

        # Determine the root node name
        # If there are multiple roots, we create a common root node
        # If there is only one root, we use that as the root node name
        # If there are no roots, we set the root node name to None

        st_name = None

        if len(roots_list) > 1:
            root_tree_node["name"] = "Model"
            tree[root_tree_node["name"]] = []
            for root_name in roots_list:
                parent[root_name] = root_tree_node["name"]
                tree[root_tree_node["name"]].append(root_name)
            st_name = root_tree_node["name"]
        elif len(roots_list) == 1:
            st_name = roots_list[0]
            root_tree_node["name"] = st_name

        # build the tree structure
        # starting from the root node
        # and traversing the tree using BFS

        if st_name is not None:
            q = deque()
            q.append(root_tree_node)
            while len(q) > 0:
                cur_node = q.popleft()
                cur_name = cur_node["name"]

                children_name = tree.get(cur_name, [])
                for child_name in children_name:
                    child_node = {
                        "name": child_name,
                        "children": [],
                        "module_type": model_type_dict.get(child_name, "Unknown"),
                    }
                    cur_node["children"].append(child_node)
                    q.append(child_node)

        return root_tree_node

    def _update_optimizer(self, cmd: Cmd, optimizer, lr_scheduler):
        """Set the learning rate for the optimizer."""

        def _update_lr(new_lr: float):
            if lr_scheduler is not None:
                logger.info("update on lr scheduler", new_lr)
                if hasattr(lr_scheduler, "base_lrs"):
                    lr_scheduler.base_lrs = [new_lr] * len(lr_scheduler.base_lrs)
                else:
                    for group in optimizer.param_groups:
                        if "lr" not in group:
                            continue
                        if isinstance(group["lr"], torch.Tensor):
                            group["lr"].fill_(new_lr)
                        else:
                            group["lr"] = new_lr

                for group in optimizer.param_groups:
                    if "initial_lr" in group:
                        if isinstance(group["initial_lr"], torch.Tensor):
                            group["initial_lr"].fill_(new_lr)
                        else:
                            group["initial_lr"] = new_lr

            elif optimizer is not None:
                logger.info("update on optimizer", new_lr)
                for param_group in optimizer.param_groups:
                    if "lr" not in param_group:
                        continue

                    if isinstance(param_group["lr"], torch.Tensor):
                        param_group["lr"].fill_(new_lr)
                    else:
                        param_group["lr"] = new_lr
            else:
                logger.warning(
                    "No optimizer or lr_scheduler found to update learning rate."
                )
                return False
            return True

        update_body_json = cmd.args
        try:
            update_body = json.loads(update_body_json)
            ret = True
            if "lr" in update_body:
                ret = ret and _update_lr(float(update_body["lr"]["value"]))

            return ret

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON format in command args: {update_body_json}")
            return False

    def _update_dataset_runtime_hyperparameter(
        self, cmd: Cmd, dataloader: torch.utils.data.DataLoader
    ):
        if not hasattr(dataloader, "dataset") or dataloader.dataset is None:
            logger.warning("Dataloader does not have a dataset attribute.")
            return False

        dataset = dataloader.dataset

        if not isinstance(dataset, InteractiveDatasetMixin):
            logger.warning("Dataset is not an instance of InteractiveDatasetMixin.")
            return False

        dataset.update_runtime_parameters(json.loads(cmd.args))

    def _build_initial_optimizer_state(self, optimizer, lr_scheduler):
        optimizer_state = {}
        if lr_scheduler:
            optimizer_state["lr"] = lr_scheduler.base_lrs[0]
        else:
            if optimizer is not None:
                cur_lr = optimizer.param_groups[0].get("lr", 0.0)
                if isinstance(cur_lr, torch.Tensor):
                    optimizer_state["lr"] = cur_lr.item()
                else:
                    optimizer_state["lr"] = cur_lr
            else:
                optimizer_state["lr"] = 0.0
        return optimizer_state

    def _build_initial_model_info(self, cur_model: torch.nn.Module):
        module_type_dict = {}
        model_info = []
        for name, module in cur_model.named_modules():
            if name.strip() == "":
                continue

            model_info.append(name)
            module_type_dict[name] = module.__class__.__name__
        return self._parse_model_tree(model_info, module_type_dict)

    def _build_initial_dataset_info(self, train_dataloader):
        all_dataset_info_event = {
            "initialization_parameters": {},
            "interactive_parameters": {},
        }

        train_dataset = train_dataloader.dataset
        if isinstance(train_dataset, InteractiveDatasetMixin):
            all_dataset_info_event["initialization_parameters"] = (
                train_dataset.get_updateable_initialization_parameters()
            )
            all_dataset_info_event["interactive_parameters"] = (
                train_dataset.get_updateable_interactive_parameters()
            )
        else:
            logger.warning(
                "Train dataset is not an instance of InteractiveDatasetMixin, "
                "cannot retrieve interactive parameters."
            )
        return all_dataset_info_event

    def on_train_begin(self, args, state, control, **kwargs):
        msg = EVENT_MESSAGE_TEMPLATE.copy()
        msg["command"] = TRAIN_STATE_UPDATE
        msg["uuid"] = str(uuid.uuid4())

        data_msg = EVENT_MESSAGE_TEMPLATE.copy()
        data_msg["command"] = INIITAL_DATASET_INFORMATION
        data_msg["uuid"] = str(uuid.uuid4())

        cur_model = kwargs.get("model", None)
        optimizer = kwargs.get("optimizer", None)
        lr_scheduler = kwargs.get("lr_scheduler", None)
        train_dataloader = kwargs.get("train_dataloader", None)
        model_info_tree = self._build_initial_model_info(cur_model)
        optimizer_state = self._build_initial_optimizer_state(optimizer, lr_scheduler)
        start_time = time.time()

        msg["time"] = start_time
        data_msg["time"] = start_time
        all_train_info = {
            "model_infos": model_info_tree,
            "optimizer_states": optimizer_state,
            "checkpoints": [],
            "command_history": [],
            "start_time": start_time,
        }

        all_dataset_info = self._build_initial_dataset_info(train_dataloader)

        msg["args"] = json.dumps(all_train_info)
        data_msg["args"] = json.dumps(all_dataset_info)

        logger.info("initial info", all_train_info)
        logger.info("initial dataset info", all_dataset_info)
        self._server_update_callback(data_msg)
        self._event_queue.put(data_msg)
        self._server_update_callback(msg)
        self._event_queue.put(msg)

    def on_step_end(self, args, state, control, **kwargs):
        while not self._cmd_queue.empty():
            cmd: Cmd = self._cmd_queue.get()
            self._cur_cmd_list.append(cmd)
            if cmd.command == UPDATE_OPTIMIZER:
                if self._update_optimizer(
                    cmd, kwargs.get("optimizer"), kwargs.get("lr_scheduler")
                ):
                    msg = {
                        "status": CMD_SUCCESS,
                        "command": UPDATE_OPTIMIZER,
                        "args": cmd.args,
                        "uuid": cmd.uuid,
                        "time": cmd.time,
                    }
                else:
                    msg = {
                        "status": CMD_SUCCESS,
                        "command": UPDATE_OPTIMIZER,
                        "args": cmd.args,
                        "uuid": cmd.uuid,
                        "time": cmd.time,
                    }

                self._server_update_callback(msg)
                self._event_queue.put(msg)

            elif cmd.command == UPDATE_DATASET_RUNTIME_HYPERPARAMETERS:
                dataloader = kwargs.get("train_dataloader", None)
                if dataloader is not None:
                    self._update_dataset_runtime_hyperparameter(cmd, dataloader)
                    msg = {
                        "status": CMD_SUCCESS,
                        "command": UPDATE_DATASET_RUNTIME_HYPERPARAMETERS,
                        "args": cmd.args,
                        "uuid": cmd.uuid,
                        "time": cmd.time,
                    }
                else:
                    msg = {
                        "status": CMD_FAILED,
                        "command": UPDATE_DATASET_RUNTIME_HYPERPARAMETERS,
                        "args": cmd.args,
                        "uuid": cmd.uuid,
                        "time": cmd.time,
                    }
                self._server_update_callback(msg)
                self._event_queue.put(msg)

            elif cmd.command == STOP_TRAINING:
                control.should_training_stop = True
                msg = {
                    "status": CMD_SUCCESS,
                    "command": STOP_TRAINING,
                    "args": cmd.args,
                    "uuid": cmd.uuid,
                    "time": cmd.time,
                }
                self._server_update_callback(msg)
                self._event_queue.put(msg)
                print("Training stopped by command.")

            elif cmd.command == LOAD_CHECKPOINT:
                control.should_training_stop = True
                msg = {
                    "status": CMD_RUNNING,
                    "command": LOAD_CHECKPOINT,
                    "args": cmd.args,
                    "uuid": cmd.uuid,
                    "time": cmd.time,
                }
                self._server_update_callback(msg)
                self._event_queue.put(msg)

            elif cmd.command == DO_EVALUATE:
                control.should_evaluate = True
                self._cmd_evaluate = True
                msg = {
                    "status": CMD_RUNNING,
                    "command": DO_EVALUATE,
                    "args": cmd.args,
                    "uuid": cmd.uuid,
                    "time": cmd.time,
                }

            elif cmd.command == RESET_LAYER:
                print("Resetting layer weights is not implemented yet.")

        return control

    def on_evaluate(self, args, state, control, **kwargs):
        if self._cmd_evaluate:
            self._cmd_evaluate = False
            msg = {
                "status": CMD_SUCCESS,
                "command": DO_EVALUATE,
                "args": "",
                "uuid": str(uuid.uuid4()),
                "time": time.time(),
            }
            self._server_update_callback(msg)
            self._event_queue.put(msg)
            control.should_evaluate = False  # Reset to avoid multiple evaluations


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
            else:
                print(f"Unknown command for stop callback: {cmd.command}")

    def on_save(self, args, state, control, **kwargs):
        """Called when a checkpoint is saved."""
        last_ckpt_dir = get_last_checkpoint(
            os.path.join(args.output_dir, self._current_branch_id_callback())
        )
        if self._is_cmd_save:
            self._is_cmd_save = False

            control.should_save = False  # Reset to avoid multiple saves
            msg = EVENT_MESSAGE_TEMPLATE.copy()
            msg["command"] = SAVE_CHECKPOINT
            msg["status"] = "success"
            self._event_queue.put(msg)

        msg_ckpt = EVENT_MESSAGE_TEMPLATE.copy()
        msg_ckpt["command"] = CHECKPOINT_INFO_UPDATE

        msg_ckpt["args"] = json.dumps(
            {
                "checkpoint_dir": last_ckpt_dir,
                "global_step": state.global_step,
                "time": os.path.getctime(last_ckpt_dir),
                "uuid": str(uuid.uuid4()),
                "branch_id": self._current_branch_id_callback(),
            }
        )
        self._server_update_callback(msg_ckpt)
        self._event_queue.put(msg_ckpt)


class LoggingCallback(InteractiveCallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_log(self, args, state, control, **kwargs):
        log = kwargs.get("logs", {})
        log["global_step"] = state.global_step
        event = self._server_update_callback(log)
        self._event_queue.put(event)


class RunPauseCallback(InteractiveCallbackBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_step_end(self, args, state, control, **kwargs):
        while not self._cmd_queue.empty():
            cmd: Cmd = self._cmd_queue.get()

            if cmd.command == PAUSE_TRAINING:
                event = {
                    "status": CMD_SUCCESS,
                    "command": PAUSE_TRAINING,
                    "args": "",
                    "uuid": "",
                    "time": 0.0,
                }
                self._server_update_callback(event)
                self._event_queue.put(event)

                while True:
                    new_cmd = (
                        self._cmd_queue.get()
                    )  # maybe block here until a command is available
                    if new_cmd.command in {RESUME_TRAINING, STOP_TRAINING}:
                        if new_cmd.command == RESUME_TRAINING:
                            new_event = {
                                "status": CMD_SUCCESS,
                                "command": RESUME_TRAINING,
                                "args": new_cmd.args,
                                "uuid": new_cmd.uuid,
                                "time": new_cmd.time,
                            }
                            self._server_update_callback(new_event)
                            self._event_queue.put(new_event)
                            print("Training resumed by command.")
                        break
            else:
                print(f"Unknown command for pause callback: {cmd.command}")

        return control
