import json
from typing import Generic
from trainer.constants import (
    WRAPER_CONTROL_COMMAND_TYPE,
    LOAD_CHECKPOINT,
)
from typing import TypeVar
from transformers import Trainer, Seq2SeqTrainer
from transformers import Trainer, TrainingArguments
from trainer.interactive_training_server import InteractiveServer
from trainer.callbacks import InteractiveCallback, CheckpointCallback, LoggingCallback

TrainerType = TypeVar("TrainerType", Trainer, Seq2SeqTrainer)


class InteractiveTrainingMixin(Generic[TrainerType]):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9876,
        timeout: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.host = host
        self.port = port
        self.timeout = timeout
        self._server = InteractiveServer(host, port, timeout)
        self._wrapper_control_queue = self._server.messages_queue_by_type[
            WRAPER_CONTROL_COMMAND_TYPE
        ]
        self._running = False

    def _set_log_integration(self, train_args: TrainingArguments):
        is_all = train_args.report_to == "all"
        if train_args.report_to == "wandb" or is_all:
            try:
                import wandb

                wandb.init(name=train_args.run_name, resume="allow")
            except ImportError:
                print("wandb is not installed. Skipping Weights & Biases integration.")
        if train_args.report_to == "tensorboard" or is_all:
            pass
        print(f"Only 'wandb' and 'tensorboard' are supported for reporting.")

    def _initialize_callback(self):

        update_callback = InteractiveCallback(
            cmd_queue=self._server.messages_queue_by_type["update"],
            event_queue=self._server.events_queue,
            server_state_update_callback=self._server.update_server_state,
        )
        checkpoint_callback = CheckpointCallback(
            cmd_queue=self._server.messages_queue_by_type["save"],
            event_queue=self._server.events_queue,
            server_state_update_callback=self._server.update_server_state,
        )

        logging_callback = LoggingCallback(
            event_queue=self._server.events_queue,
            server_state_update_callback=self._server.update_log_state,
        )
        self.add_callback(update_callback)
        self.add_callback(checkpoint_callback)
        self.add_callback(logging_callback)

    def train(self, **kwargs):
        """
        Start the training process with the Trainer instance.
        This method will run the training loop and listen for commands
        from the InteractiveServer.
        """
        self._server.run()
        self._initialize_callback()

        while True:
            super().train(**kwargs)

            print("Train exited ... waiting for commands")

            # Check if we have a load_checkpoint command
            if self._wrapper_control_queue.empty():
                print("Empty wrapper control queue, waiting for commands...")
                break

            is_load = False
            load_config = None
            while not self._wrapper_control_queue.empty():
                cmd = self._wrapper_control_queue.get()
                if cmd.command == LOAD_CHECKPOINT:
                    self._load_message = cmd
                    print(f"Received load_checkpoint command: {cmd}")
                    is_load = True
                    load_config = json.loads(cmd.args)

            if is_load:
                self._load_message = cmd
                print(f"Received load_checkpoint command: {cmd}")
                print(f"load config: {load_config}")

                ckpt_info = self._server.get_checkpoint_info(load_config["uuid"])
                if ckpt_info is None:
                    print(f"Checkpoint with UUID {load_config['uuid']} not found.")
                    break

                kwargs["resume_from_checkpoint"] = ckpt_info["checkpoint_dir"]

            else:
                break

        self._server.stop()
