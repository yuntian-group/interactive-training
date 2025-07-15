from typing import Optional
from pydantic import BaseModel

LOAD_CHECKPOINT = "load_checkpoint"
SAVE_CHECKPOINT = "save_checkpoint"
STOP_TRAINING = "stop_training"
UPDATE_OPTIMIZER = "update_optimizer"

# deprecated
RESET_LAYER = "reset_layer"

MODEL_LAYER_OPERATION = "model_layer_operation"
MODEL_LAYER_PARAMETER_UPDATE = "model_layer_parameter_update"
DO_EVALUATE = "do_evaluate"
WRAPPER_RESUME_FOR_LOAD_CHECKPOINT = "wrapper_resume_load_checkpoint"
TRAIN_STATE_UPDATE = "train_state_update"
CHECKPOINT_INFO_UPDATE = "checkpoint_info_update"
LOG_UPDATE = "log_update"
PAUSE_TRAINING = "pause_training"
RESUME_TRAINING = "resume_training"

INIITAL_DATASET_INFORMATION = "initial_dataset_information"
UPDATE_DATASET = "update_dataset"
UPDATE_DATASET_RUNTIME_HYPERPARAMETERS = "update_dataset_runtime_hyperparameters"

UPDATE_COMMANDS = {
    UPDATE_OPTIMIZER,
    SAVE_CHECKPOINT,
    RESET_LAYER,
    DO_EVALUATE,
    MODEL_LAYER_OPERATION,
    MODEL_LAYER_PARAMETER_UPDATE,
}
LOAD_COMMANDS = {LOAD_CHECKPOINT}
CONTROL_COMMANDS = {STOP_TRAINING}
WRAPPER_CONTROL_COMMANDS = {WRAPPER_RESUME_FOR_LOAD_CHECKPOINT}
PAUSE_RESUME_COMMANDS = {PAUSE_TRAINING, RESUME_TRAINING}


UPDATE_COMMAND_TYPE = "update"
CONTROL_COMMAND_TYPE = "control"
WRAPER_CONTROL_COMMAND_TYPE = "wrapper_control"
LOAD_COMMAND_TYPE = "load"
SAVE_COMMAND_TYPE = "save"
PAUSE_RESUME_TYPE = "pause_resume"
DATASET_RELOAD_TYPE = "dataset_reload"


COMMAND_TO_TYPE = {
    UPDATE_OPTIMIZER: UPDATE_COMMAND_TYPE,
    SAVE_CHECKPOINT: SAVE_COMMAND_TYPE,
    RESET_LAYER: UPDATE_COMMAND_TYPE,
    DO_EVALUATE: UPDATE_COMMAND_TYPE,
    LOAD_CHECKPOINT: UPDATE_COMMAND_TYPE,
    STOP_TRAINING: UPDATE_COMMAND_TYPE,
    MODEL_LAYER_PARAMETER_UPDATE: UPDATE_COMMAND_TYPE,
    MODEL_LAYER_OPERATION: UPDATE_COMMAND_TYPE,
    UPDATE_DATASET_RUNTIME_HYPERPARAMETERS: UPDATE_COMMAND_TYPE,
    WRAPPER_RESUME_FOR_LOAD_CHECKPOINT: WRAPER_CONTROL_COMMAND_TYPE,
    PAUSE_TRAINING: PAUSE_RESUME_TYPE,
    RESUME_TRAINING: PAUSE_RESUME_TYPE,
    UPDATE_DATASET: DATASET_RELOAD_TYPE,
}

CMD_REQUESTED = "requested"
CMD_PENDING = "pending"
CMD_RUNNING = "running"
CMD_COMPELTED = "completed"
CMD_SUCCESS = "success"
CMD_FAILED = "failed"

MAIN_BRANCH_NAME = "main"


EVENT_MESSAGE_TEMPLATE = {
    "status": CMD_SUCCESS,
    "command": "",
    "args": "",
    "uuid": "",
    "time": 0.0,
}

OPTIMIZER_TUNABLE_PARAMETER_NAMES = {
    "torch.optim.sgd.SGD": {
        "tunable_parameters": [
            {"name": "lr", "type": "float"},
            {"name": "momentum", "type": "float"},
            {"name": "dampening", "type": "float"},
            {"name": "nesterov", "type": "bool"},
        ],
        "update_type": "param_groups",
    },
    "torch.optim.adam.Adam": {
        "tunable_parameters": [
            {"name": "lr", "type": "float"},
            {"name": "betas", "type": "tuple"},
            {"name": "eps", "type": "float"},
            {"name": "weight_decay", "type": "float"},
            {"name": "amsgrad", "type": "bool"},
        ],
        "update_type": "param_groups",
    },
    "torch.optim.adamw.AdamW": {
        "tunable_parameters": [
            {"name": "lr", "type": "float"},
            {"name": "betas", "type": "tuple"},
            {"name": "eps", "type": "float"},
            {"name": "weight_decay", "type": "float"},
            {"name": "amsgrad", "type": "bool"},
            {"name": "decoupled_weight_decay", "type": "bool"},
        ],
        "update_type": "param_groups",
    },
    "torch.optim.rmsprop.RMSprop": {
        "tunable_parameters": [
            {"name": "lr", "type": "float"},
            {"name": "eps", "type": "float"},
            {"name": "weight_decay", "type": "float"},
            {"name": "momentum", "type": "float"},
            {"name": "centered", "type": "bool"},
        ],
        "update_type": "param_groups",
    },
    # TO BE ADDED
}

DEFAULT_TUNABLE_PARAMETER_NAMES = {
    "tunable_parameters": [{"name": "lr", "type": "float"}],
    "update_type": "param_groups",
}


DEFAULT_TUNEABLE_MODULE_PARAMETERS = {"p", "eps", "dropout"}

DEFAULT_TUNEABLE_MODULE_OPERATIONS = {"reset_parameters", "requires_grad_"}


TYPE_MAPPING = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "tuple": tuple,
    "set": set,
}

BASIC_TYPES = (int, float, str, bool)
NESTED_TYPES = (list, dict, tuple, set)
RESERVED_NAMES = {
    "_cmd_queue",
    "_control_queue",
    "_interactive_parameter_names",
    "_interactive_parameters",
    "_dataset_kwargs",
}


class Cmd(BaseModel):
    command: str
    args: Optional[str] = None
    time: float
    uuid: str
    status: str
