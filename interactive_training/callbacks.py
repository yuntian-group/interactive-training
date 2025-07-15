import os
import time
import uuid
import json
import queue
import torch
import inspect
from collections import deque
from transformers.utils import logging
from transformers import TrainerCallback
from typing import Callable, List, Dict
from accelerate.optimizer import AcceleratedOptimizer
from transformers.trainer_utils import get_last_checkpoint
from interactive_training.utils import get_full_optimizer_type_name
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
    OPTIMIZER_TUNABLE_PARAMETER_NAMES,
    DEFAULT_TUNABLE_PARAMETER_NAMES,
    DEFAULT_TUNEABLE_MODULE_OPERATIONS,
    DEFAULT_TUNEABLE_MODULE_PARAMETERS,
    MODEL_LAYER_OPERATION,
    MODEL_LAYER_PARAMETER_UPDATE,
    BASIC_TYPES,
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

    def _parse_model_tree(self, model_info_dict: dict):
        """Parse a list of module names into a tree structure."""
        tree = {}
        parent = {}
        module_names_split = list(
            sorted(
                [name.split(".") for name in model_info_dict.keys()],
                key=lambda x: len(x),
            )
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
        for name in model_info_dict:
            if name not in parent:
                roots_list.append(name)

        root_info = model_info_dict.get(name, {})

        root_tree_node = {
            "name": None,
            "children": [],
            "module_type": root_info.get("module_type", "Unknown"),
            "operators": root_info.get("operators", []),
            "hyperparameters": root_info.get("hyperparameters", {}),
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
                    cur_child_info = model_info_dict.get(child_name, {})
                    child_node = {
                        "name": child_name,
                        "children": [],
                        "module_type": cur_child_info["module_type"],
                        "operators": cur_child_info["operators"],
                        "hyperparameters": cur_child_info.get("hyperparameters", {}),
                    }
                    cur_node["children"].append(child_node)
                    q.append(child_node)

        return root_tree_node

    def _update_optimizer(self, cmd: Cmd, optimizer_raw, lr_scheduler):
        """Set the learning rate for the optimizer."""

        if optimizer_raw is None:
            logger.warning("No optimizer found to update parameters.")
            return False

        if isinstance(optimizer_raw, AcceleratedOptimizer):
            optimizer = optimizer_raw.optimizer
        else:
            optimizer = optimizer_raw

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

        def _update_param(
            update_value: Dict[str, any], update_type: str = "param_groups"
        ):
            if update_type == "param_groups":
                if optimizer is None:
                    logger.warning("No optimizer found to update parameters.")
                    return False

                for param_group in optimizer.param_groups:
                    for key, value in update_value.items():
                        if key in param_group:
                            if isinstance(param_group[key], torch.Tensor):
                                param_group[key].fill_(value)
                            else:
                                param_group[key] = value
                        else:
                            param_group[key] = value
                            logger.warning(
                                f"Parameter {key} not found in optimizer group, "
                                "adding it with value {value}."
                            )
                return True
            else:
                logger.warning(f"Update type {update_type} is not implemented.")
                return False

        update_body_json = cmd.args
        tunable_parameters_configs = OPTIMIZER_TUNABLE_PARAMETER_NAMES.get(
            get_full_optimizer_type_name(optimizer), DEFAULT_TUNABLE_PARAMETER_NAMES
        )

        tunable_parameters = tunable_parameters_configs.get("tunable_parameters", [])
        update_type = tunable_parameters_configs.get("update_type", "param_groups")

        all_tunable_parameters_dict = {
            param["name"]: param for param in tunable_parameters
        }

        try:
            update_body = json.loads(update_body_json)
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON format in command args: {update_body_json}")
            return False

        ret = True
        bulk_update_params = {}
        print("Update body:", update_body)
        for param_name, param_value in update_body.items():
            if param_name == "lr":
                ret = ret and _update_lr(float(update_body["lr"]["value"]))
            elif param_name in all_tunable_parameters_dict:
                param_config = all_tunable_parameters_dict[param_name]
                if param_config["type"] == "float":
                    bulk_update_params[param_name] = float(param_value["value"])
                elif param_config["type"] == "bool":
                    bulk_update_params[param_name] = bool(param_value["value"])
                elif param_config["type"] == "tuple":
                    bulk_update_params[param_name] = tuple(
                        float(v) for v in param_value["value"]
                    )
                else:
                    logger.warning(
                        f"Unsupported parameter type {param_config['type']} for {param_name}."
                    )
            else:
                logger.warning(
                    f"Parameter {param_name} is not a tunable parameter for the optimizer."
                )
        if len(bulk_update_params) > 0:
            ret = ret and _update_param(bulk_update_params, update_type)
        else:
            if ("lr" in update_body and len(update_body) == 1) or len(update_body) == 0:
                logger.warning(
                    "No valid tunable parameters found in the update command. "
                    "Ensure the command contains valid parameters."
                )

        return ret

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

    def _build_initial_optimizer_state(self, optimizer_raw, lr_scheduler):

        if optimizer_raw is None:
            logger.warning("No optimizer found to update parameters.")
            return False

        if isinstance(optimizer_raw, AcceleratedOptimizer):
            optimizer = optimizer_raw.optimizer
        else:
            optimizer = optimizer_raw

        def _get_lr():
            cur_lr = 0.0
            if lr_scheduler:
                cur_lr = lr_scheduler.base_lrs[0]
            else:
                if optimizer is not None:
                    cur_lr = optimizer.param_groups[0].get("lr", 0.0)
                    if isinstance(cur_lr, torch.Tensor):
                        cur_lr = cur_lr.item()
                    else:
                        cur_lr = cur_lr
                else:
                    cur_lr = 0.0
            return cur_lr

        def _get_param_from_optimizer_group(
            name: str, update_type: str = "param_groups"
        ):
            if optimizer is None:
                return None

            if update_type == "param_groups":
                cur_group = optimizer.param_groups[0]
                if name in cur_group:
                    value = cur_group[name]
                    if isinstance(value, torch.Tensor):
                        return value.item()
                    return value
                else:
                    logger.warning(f"Parameter {name} not found in optimizer group.")
                    return None
            else:
                raise NotImplementedError(
                    f"Update type {update_type} is not implemented."
                )

        optimizer_state = {}
        optimizer_full_name = get_full_optimizer_type_name(optimizer)

        optimizer_state["optimizer_name"] = optimizer_full_name

        tune_tmp = OPTIMIZER_TUNABLE_PARAMETER_NAMES.get(
            optimizer_full_name, DEFAULT_TUNABLE_PARAMETER_NAMES
        )
        cur_optimizer_tunable_parameters = tune_tmp.get("tunable_parameters", [])

        for tunable_param in cur_optimizer_tunable_parameters:
            if tunable_param["name"] == "lr":
                optimizer_state["lr"] = _get_lr()
            else:
                tmp_value = _get_param_from_optimizer_group(tunable_param["name"])
                if tmp_value is not None:
                    optimizer_state[tunable_param["name"]] = tmp_value

        return optimizer_state

    def _build_module_tuneable_parameters(
        self, cur_module: torch.nn.Module, candidate_set: set[str]
    ) -> Dict[str, any]:
        all_params = dir(cur_module)
        tuneable_params = []
        for param in all_params:
            if param in candidate_set:
                value = getattr(cur_module, param, None)
                if value is not None and isinstance(value, BASIC_TYPES):
                    tuneable_params.append(
                        {
                            "name": param,
                            "value": value,
                            "type": type(value).__name__,
                        }
                    )
                else:
                    logger.warning(
                        f"Unsupported type for parameter {param}: {type(value)}"
                    )
        return tuneable_params

    def _build_module_tuneable_operations(
        self, cur_module: torch.nn.Module, candidate_set: set[str]
    ) -> List[str]:
        all_opereations = dir(cur_module)
        callable_operations = []
        for operation in all_opereations:
            if operation in candidate_set:
                value = getattr(cur_module, operation, None)
                if callable(value):
                    sig = inspect.signature(value)
                    params = []
                    for param_name, param in sig.parameters.items():
                        param_info = {
                            "name": param_name,
                            "type": (
                                param.annotation.__name__
                                if param.annotation != inspect.Parameter.empty
                                else ""
                            ),
                            "value": (
                                str(param.default)
                                if param.default != inspect.Parameter.empty
                                else ""
                            ),
                        }
                        params.append(param_info)

                    callable_operations.append({"name": operation, "signature": params})
                else:
                    logger.warning(
                        f"Operation {operation} is not callable or not found in module."
                    )
        return callable_operations

    def _build_initial_model_info(self, cur_model: torch.nn.Module):
        # module_type_dict = {}
        model_info_dict = {}
        for name, module in cur_model.named_modules():
            if name.strip() == "":
                continue

            # model_names.append(name)
            # module_type_dict[name] = module.__class__.__name__

            cur_module_info = {
                "name": name,
                "module_type": module.__class__.__name__,
                "operators": self._build_module_tuneable_operations(
                    module, DEFAULT_TUNEABLE_MODULE_OPERATIONS
                ),
                "hyperparameters": self._build_module_tuneable_parameters(
                    module, DEFAULT_TUNEABLE_MODULE_PARAMETERS
                ),
            }

            model_info_dict[name] = cur_module_info

        return self._parse_model_tree(model_info_dict)

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

    def _update_layer_parameters(self, cmd: Cmd, model: torch.nn.Module):

        print("Updating layer parameters with command:", cmd)

        args = json.loads(cmd.args)
        layer_name = args.get("layer_name", None)
        param_name = args.get("param_name", None)
        value = args.get("value", None)
        if layer_name is None or param_name is None or value is None:
            logger.warning(
                "Layer name, parameter name, or value is not provided in the command."
            )
            return False

        for name, module in model.named_modules():
            if name == layer_name:
                if hasattr(module, param_name):
                    param_value = getattr(module, param_name)
                    if isinstance(param_value, torch.Tensor):
                        param_value.fill_(value)
                    else:
                        setattr(module, param_name, value)
                    logger.info(
                        f"Updated parameter {param_name} of layer {layer_name} to {value}."
                    )
                    return True
                else:
                    return False

        logger.warning(f"Layer {layer_name} not found in the model.")
        return False

    def _perform_layer_operation(self, cmd: Cmd, model: torch.nn.Module):

        print("Performing layer operation with command:", cmd)

        args = json.loads(cmd.args)
        layer_name = args.get("layer_name", None)
        operation_name = args.get("operation_name", None)
        params = args.get("params", {})
        if layer_name is None or operation_name is None:
            logger.warning(
                "Layer name or operation name is not provided in the command."
            )
            return False
        for name, module in model.named_modules():
            if name == layer_name:
                if hasattr(module, operation_name):
                    operation = getattr(module, operation_name)
                    if callable(operation):
                        try:
                            if len(params) > 0:
                                operation(**params)
                            else:
                                operation()
                            logger.info(
                                f"Performed operation {operation_name} on layer {layer_name}."
                            )
                            return True
                        except Exception as e:
                            logger.error(
                                f"Error performing operation {operation_name} on layer {layer_name}: {e}"
                            )
                            return False
                    else:
                        logger.warning(
                            f"{operation_name} is not callable in layer {layer_name}."
                        )
                        return False
                else:
                    logger.warning(
                        f"Operation {operation_name} not found in layer {layer_name}."
                    )
                    return False
        logger.warning(f"Layer {layer_name} not found in the model.")
        return False

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
            "run_name": (
                args.run_name
                if args.run_name is not None
                else "Interactive Hugging Face Trainer"
            ),
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

            elif cmd.command == MODEL_LAYER_OPERATION:
                if self._perform_layer_operation(cmd, kwargs.get("model", None)):
                    msg = {
                        "status": CMD_SUCCESS,
                        "command": MODEL_LAYER_OPERATION,
                        "args": cmd.args,
                        "uuid": cmd.uuid,
                        "time": cmd.time,
                    }
                else:
                    msg = {
                        "status": CMD_FAILED,
                        "command": MODEL_LAYER_OPERATION,
                        "args": cmd.args,
                        "uuid": cmd.uuid,
                        "time": cmd.time,
                    }
                self._server_update_callback(msg)
                self._event_queue.put(msg)

            elif cmd.command == MODEL_LAYER_PARAMETER_UPDATE:
                if self._update_layer_parameters(cmd, kwargs.get("model", None)):
                    msg = {
                        "status": CMD_SUCCESS,
                        "command": MODEL_LAYER_PARAMETER_UPDATE,
                        "args": cmd.args,
                        "uuid": cmd.uuid,
                        "time": cmd.time,
                    }
                else:
                    msg = {
                        "status": CMD_FAILED,
                        "command": MODEL_LAYER_PARAMETER_UPDATE,
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
