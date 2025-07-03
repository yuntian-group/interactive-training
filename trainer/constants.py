from typing import Optional
from pydantic import BaseModel

LOAD_CHECKPOINT = "load_checkpoint"
SAVE_CHECKPOINT = "save_checkpoint"
STOP_TRAINING = "stop_training"
UPDATE_OPTIMIZER = "update_optimizer"
RESET_LAYER = "reset_layer"
DO_EVALUATE = "do_evaluate"
WRAPPER_RESUME_FOR_LOAD_CHECKPOINT = "wrapper_resume_load_checkpoint"
TRAIN_STATE_UPDATE = "train_state_update"
CHECKPOINT_INFO_UPDATE = "checkpoint_info_update"
LOG_UPDATE = "log_update"
PAUSE_TRAINING = "pause_training"
RESUME_TRAINING = "resume_training"

UPDATE_COMMANDS = {UPDATE_OPTIMIZER, SAVE_CHECKPOINT, RESET_LAYER, DO_EVALUATE}
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
SUCCESS = "success"


COMMAND_TO_TYPE = {
    UPDATE_OPTIMIZER: UPDATE_COMMAND_TYPE,
    SAVE_CHECKPOINT: SAVE_COMMAND_TYPE,
    RESET_LAYER: UPDATE_COMMAND_TYPE,
    DO_EVALUATE: UPDATE_COMMAND_TYPE,
    LOAD_CHECKPOINT: UPDATE_COMMAND_TYPE,
    STOP_TRAINING: CONTROL_COMMAND_TYPE,
    WRAPPER_RESUME_FOR_LOAD_CHECKPOINT: WRAPER_CONTROL_COMMAND_TYPE,
    PAUSE_TRAINING: PAUSE_RESUME_TYPE,
    RESUME_TRAINING: PAUSE_RESUME_TYPE,
}

CMD_REQUESTED = "requested"
CMD_PENDING = "pending"
CMD_RUNNING = "running"
CMD_COMPELTED = "completed"
CMD_SUCCESS = "success"
CMD_FAILED = "failed"

MAIN_BRANCH_NAME = "main"


EVENT_MESSAGE_TEMPLATE = {
    "status": SUCCESS,
    "command": "",
    "args": "",
    "uuid": "",
    "time": 0.0,
}


class Cmd(BaseModel):
    command: str
    args: Optional[str] = None
    time: float
    uuid: str
    status: str
