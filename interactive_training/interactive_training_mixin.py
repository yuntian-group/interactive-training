import os
import time
import json
import torch
import shutil
import functools
import contextlib
from torch import nn
import torch.distributed as dist
from typing import Generic, TypeVar, Optional
from interactive_training.constants import (
    WRAPER_CONTROL_COMMAND_TYPE,
    DATASET_RELOAD_TYPE,
    LOAD_CHECKPOINT,
    STOP_TRAINING,
    UPDATE_DATASET,
    CMD_SUCCESS,
    CMD_FAILED,
)
from interactive_training.interactive_dataset_mixin import InteractiveDatasetMixin

from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.integrations.deepspeed import (
    deepspeed_init,
    deepspeed_load_checkpoint,
)
from transformers.trainer_utils import speed_metrics, TrainOutput
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer import _is_peft_model, TRAINER_STATE_NAME
from transformers.trainer_callback import ProgressCallback
from transformers.modeling_utils import unwrap_model
from transformers.training_args import OptimizerNames, ParallelMode
from transformers.trainer_callback import ExportableState, TrainerState
from transformers.utils import (
    is_sagemaker_mp_enabled,
    is_torch_xla_available,
    is_accelerate_available,
    logging,
)
from transformers import Trainer, Seq2SeqTrainer, TrainingArguments
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from interactive_training.interactive_training_server import InteractiveServer
from interactive_training.callbacks import (
    InteractiveCallback,
    CheckpointCallback,
    LoggingCallback,
    RunPauseCallback,
)


if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate.utils import DistributedType

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

logger = logging.get_logger(__name__)

TrainerType = TypeVar("TrainerType", Trainer, Seq2SeqTrainer)


class InteractiveTrainingMixin(Generic[TrainerType]):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7007,
        timeout: int = 10,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.host = host
        self.port = port
        self.timeout = timeout
        self._server = InteractiveServer(host, port, timeout)
        self._running = False

        self._mixin_control_queue = self._server.messages_queue_by_type[
            WRAPER_CONTROL_COMMAND_TYPE
        ]
        self._dataset_reload_control_queue = self._server.messages_queue_by_type[
            DATASET_RELOAD_TYPE
        ]

    def _set_log_integration(self, train_args: TrainingArguments):
        is_all = train_args.report_to == "all"
        if train_args.report_to == "wandb" or is_all:
            try:
                import wandb

                wandb.init(name=train_args.run_name, resume="allow")
            except ImportError:
                logger.warning(
                    "wandb is not installed. Skipping Weights & Biases integration."
                )
        logger.info(f"Only 'wandb' are supported for reporting.")

    def _initialize_callback(self):

        update_callback = InteractiveCallback(
            cmd_queue=self._server.messages_queue_by_type["update"],
            event_queue=self._server.events_queue,
            current_branch_id_callback=self._server.get_current_branch,
            server_state_update_callback=self._server.update_server_state,
        )

        checkpoint_callback = CheckpointCallback(
            cmd_queue=self._server.messages_queue_by_type["save"],
            event_queue=self._server.events_queue,
            current_branch_id_callback=self._server.get_current_branch,
            server_state_update_callback=self._server.update_server_state,
        )

        logging_callback = LoggingCallback(
            event_queue=self._server.events_queue,
            current_branch_id_callback=self._server.get_current_branch,
            server_state_update_callback=self._server.update_log_state,
        )

        pause_resume_callback = RunPauseCallback(
            cmd_queue=self._server.messages_queue_by_type["pause_resume"],
            event_queue=self._server.events_queue,
            current_branch_id_callback=self._server.get_current_branch,
            server_state_update_callback=self._server.update_server_state,
        )

        # callbacks executed in order
        self.add_callback(update_callback)
        self.add_callback(checkpoint_callback)
        self.add_callback(logging_callback)
        self.add_callback(pause_resume_callback)

    def _get_output_dir(self, trial):
        """
        Override the output directory to use the server's output directory.
        This is necessary to ensure that the Trainer uses the correct directory
        for saving checkpoints and logs.
        """
        original_output_dir = super()._get_output_dir(trial)
        branched_ckpt_dir = os.path.join(
            original_output_dir, self._server.get_current_branch()
        )

        if not os.path.exists(branched_ckpt_dir):
            os.makedirs(branched_ckpt_dir, exist_ok=True)

        logger.info(f"Using branched checkpoint directory: {branched_ckpt_dir}")

        return str(branched_ckpt_dir)

    def _maybe_reload_train_dataset(self, train_dataloader):

        cur_cmds = []
        reload_params = []

        while not self._dataset_reload_control_queue.empty():
            command = self._dataset_reload_control_queue.get()
            if command.command == UPDATE_DATASET:
                updated_params = json.loads(command.args)
                reload_params.append(updated_params)
                cur_cmds.append(command)
            else:
                logger.warning(
                    f"Received unknown command: {command.command} in dataset reload queue."
                )

        if len(cur_cmds) == 0:
            return train_dataloader, False

        if not isinstance(self.train_dataset, InteractiveDatasetMixin):
            logger.warning(
                "The train_dataset is not an instance of InteractiveDatasetMixin. "
                "Interactive dataset reload functionality will not be available."
            )
            for cmd in cur_cmds:
                self._server.enqueue_event(
                    {
                        "status": "failed",
                        "command": cmd.command,
                        "args": cmd.args,
                        "uuid": cmd.uuid,
                        "time": cmd.time,
                    }
                )
            ret = train_dataloader, False
        else:
            success_flags = []
            for i in range(len(cur_cmds)):
                success = self.train_dataset.update_intialization_parameters(
                    reload_params[i]
                )
                success_flags.append(success)

            self.train_dataset.reload_dataset()
            ret = self.get_train_dataloader(), True

            for i, cmd in enumerate(cur_cmds):
                cur_event = {
                    "status": CMD_SUCCESS if success_flags[i] else CMD_FAILED,
                    "command": cmd.command,
                    "args": cmd.args,
                    "uuid": cmd.uuid,
                    "time": cmd.time,
                }

                if success_flags[i]:
                    self._server.update_dataset_state(cur_event)
                    self._server.enqueue_event(cur_event)
                else:
                    self._server.enqueue_event(cur_event)

        return ret

    def _inner_training_loop(
        self,
        batch_size=None,
        args=None,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the initial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = (
                        self._train_batch_size // max(1, self.args.n_gpu)
                    )
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(
            f"Currently training with a batch size of: {self._train_batch_size}"
        )
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self.get_total_train_batch_size(args)

        (
            num_train_epochs,
            num_update_steps_per_epoch,
            num_examples,
            num_train_samples,
            epoch_based,
            len_dataloader,
            max_steps,
        ) = self.set_initial_training_values(
            args, train_dataloader, total_train_batch_size
        )

        num_train_tokens = None
        if self.args.include_tokens_per_second:
            num_train_tokens = self.num_tokens(
                train_dataloader, None if epoch_based else max_steps
            )
            # If going by epochs, multiply tokens linearly
            if len_dataloader is not None and epoch_based:
                num_train_tokens *= args.num_train_epochs
            # Otherwise since its steps, we just multiply by grad accum
            else:
                num_train_tokens *= args.gradient_accumulation_steps

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            is_sagemaker_mp_enabled()
            or self.is_fsdp_xla_enabled
            or self.is_fsdp_enabled
            or self.is_tp_enabled
        )

        # Can't delay optimizer creation when using FSDP2: https://github.com/huggingface/accelerate/blob/3f636d626063ffcf9a337c7d3624d61b7d187d59/src/accelerate/accelerator.py#L1404
        is_fsdp2 = self.is_fsdp_enabled and (
            getattr(self.accelerator.state.fsdp_plugin, "fsdp_version", 1) == 2
        )
        if is_fsdp2:
            delay_optimizer_creation = False

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps
            )

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[
                cb
                for cb in self.callback_handler.callbacks + [self.control]
                if isinstance(cb, ExportableState)
            ]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        self.state.compute_steps(args, max_steps)

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=args.gradient_checkpointing_kwargs
            )

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if use_accelerator_prepare and self.is_fsdp_enabled:
            # In case of auto_find_batch_size=True
            # Remove FSDP wrapping from sub-models.
            self.model = unwrap_model(self.model, recursive=True)

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                # configure fsdp plugin for qlora if any
                self._fsdp_qlora_plugin_updates()
                if self.accelerator.mixed_precision != "fp8":
                    self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    if delay_optimizer_creation:
                        self.optimizer = self.accelerator.prepare(self.optimizer)
                    else:
                        model, self.optimizer = self.accelerator.prepare(
                            self.model, self.optimizer
                        )
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped,
                    resume_from_checkpoint,
                    load_module_strict=not _is_peft_model(self.model),
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        self._load_scaler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(
            f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}"
        )
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(
                f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}"
            )
        logger.info(
            f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}"
        )
        logger.info(
            f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}"
        )
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(
            f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = int(self.state.global_step // num_update_steps_per_epoch)
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (
                    num_update_steps_per_epoch
                )
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info(
                "  Continuing training from checkpoint, will skip to saved global_step"
            )
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(
                f"  Continuing training from global step {self.state.global_step}"
            )
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        for attr in ("model", "optimizer", "lr_scheduler"):
            setattr(self.callback_handler, attr, getattr(self, attr))
        self.callback_handler.train_dataloader = train_dataloader

        self.state.init_training_references(self, max_steps, num_train_epochs, trial)

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0, device=args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        learning_rate = None
        self.control = self.callback_handler.on_train_begin(
            args, self.state, self.control
        )

        if args.eval_on_start:
            self._evaluate(trial, ignore_keys_for_eval, skip_scheduler=True)

        epoch = epochs_trained

        while epoch < num_train_epochs:

            # Reload dataloader logic start

            train_dataloader, is_reload = self._maybe_reload_train_dataset(
                train_dataloader
            )

            if is_reload:
                if self.is_fsdp_xla_v2_enabled:
                    train_dataloader = tpu_spmd_dataloader(train_dataloader)

                (
                    num_train_epochs,
                    num_update_steps_per_epoch,
                    num_examples,
                    num_train_samples,
                    epoch_based,
                    len_dataloader,
                    max_steps,
                ) = self.set_initial_training_values(
                    args, train_dataloader, total_train_batch_size
                )

                self.callback_handler.train_dataloader = train_dataloader
                self.state.init_training_references(
                    self, max_steps, num_train_epochs, trial
                )

                for cb in self.callback_handler.callbacks:
                    if hasattr(cb, "train_dataloader"):
                        cb.train_dataloader = train_dataloader

                    if isinstance(cb, ProgressCallback):
                        cb.training_bar.total = self.state.max_steps

            # Reload dataloader logic end

            epoch_dataloader = train_dataloader

            if hasattr(epoch_dataloader, "set_epoch"):
                epoch_dataloader.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_dataloader)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(
                args, self.state, self.control
            )

            if (
                epoch == epochs_trained
                and resume_from_checkpoint is not None
                and steps_trained_in_current_epoch == 0
            ):
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_dataloader = skip_first_batches(
                    epoch_dataloader, steps_trained_in_current_epoch
                )
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            epoch_iterator = iter(epoch_dataloader)
            # We chunkify the epoch iterator into gradient accumulation steps `n` batches
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0:
                remainder = args.gradient_accumulation_steps
            update_step = -1
            total_updates = steps_in_epoch // args.gradient_accumulation_steps + int(
                remainder < args.gradient_accumulation_steps
            )
            for _ in range(total_updates):
                update_step += 1
                num_batches = (
                    args.gradient_accumulation_steps
                    if update_step != (total_updates - 1)
                    else remainder
                )
                batch_samples, num_items_in_batch = self.get_batch_samples(
                    epoch_iterator, num_batches, args.device
                )
                for i, inputs in enumerate(batch_samples):
                    step += 1
                    do_sync_step = (
                        step + 1
                    ) % args.gradient_accumulation_steps == 0 or (
                        step + 1
                    ) == steps_in_epoch
                    # Since we perform prefetching, we need to manually set sync_gradients
                    self.accelerator.gradient_state._set_sync_gradients(do_sync_step)

                    if self.args.include_num_input_tokens_seen:
                        main_input_name = getattr(
                            self.model, "main_input_name", "input_ids"
                        )
                        if main_input_name not in inputs:
                            logger.warning(
                                "Tried to track the number of tokens seen, however the current model is "
                                "not configured properly to know what item is the input. To fix this, add "
                                "a `main_input_name` attribute to the model class you are using."
                            )
                        else:
                            input_tokens = inputs[main_input_name].numel()
                            input_tokens = torch.tensor(
                                input_tokens, device=self.args.device, dtype=torch.int64
                            )
                            self.state.num_input_tokens_seen += (
                                self.accelerator.gather(input_tokens).sum().item()
                            )
                    if rng_to_sync:
                        self._load_rng_state(resume_from_checkpoint)
                        rng_to_sync = False

                    # Skip past any already trained steps if resuming training
                    if steps_trained_in_current_epoch > 0:
                        steps_trained_in_current_epoch -= 1
                        if steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.update(1)
                        if steps_trained_in_current_epoch == 0:
                            self._load_rng_state(resume_from_checkpoint)
                        continue
                    elif steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.close()
                        steps_trained_progress_bar = None

                    if step % args.gradient_accumulation_steps == 0:
                        self.control = self.callback_handler.on_step_begin(
                            args, self.state, self.control
                        )

                    # We explicitly want to avoid relying on `accelerator.accumulate` for generation training
                    context = (
                        functools.partial(self.accelerator.no_sync, model=model)
                        if i != len(batch_samples) - 1
                        and self.accelerator.distributed_type
                        != DistributedType.DEEPSPEED
                        else contextlib.nullcontext
                    )
                    with context():
                        tr_loss_step = self.training_step(
                            model, inputs, num_items_in_batch
                        )

                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_xla_available()
                        and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        tr_loss = tr_loss + tr_loss / (
                            1 + self.state.global_step - self._globalstep_last_logged
                        )
                    else:
                        if tr_loss.device != tr_loss_step.device:
                            raise ValueError(
                                f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                            )
                        tr_loss = tr_loss + tr_loss_step

                    self.current_flos += float(self.floating_point_ops(inputs))

                    if do_sync_step:
                        # Since we perform prefetching, we need to manually set sync_gradients to True
                        self.accelerator.gradient_state._set_sync_gradients(True)

                        # Gradient clipping
                        if args.max_grad_norm is not None and args.max_grad_norm > 0:
                            if is_sagemaker_mp_enabled() and args.fp16:
                                _grad_norm = self.optimizer.clip_master_grads(
                                    args.max_grad_norm
                                )
                            elif self.use_apex:
                                from apex import amp

                                # Revert to normal clipping otherwise, handling Apex or full precision
                                _grad_norm = nn.utils.clip_grad_norm_(
                                    amp.master_params(self.optimizer),
                                    args.max_grad_norm,
                                )
                            else:
                                grad_norm_context = contextlib.nullcontext
                                if self.is_tp_enabled:
                                    from torch.distributed._tensor.experimental import (
                                        implicit_replication,
                                    )

                                    grad_norm_context = implicit_replication
                                with grad_norm_context():
                                    _grad_norm = self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )

                            if (
                                is_accelerate_available()
                                and self.accelerator.distributed_type
                                == DistributedType.DEEPSPEED
                            ):
                                grad_norm = model.get_global_grad_norm()
                                # In some cases the grad norm may not return a float
                                if hasattr(grad_norm, "item"):
                                    grad_norm = grad_norm.item()
                            else:
                                grad_norm = _grad_norm

                        self.control = self.callback_handler.on_pre_optimizer_step(
                            args, self.state, self.control
                        )

                        self.optimizer.step()

                        self.control = self.callback_handler.on_optimizer_step(
                            args, self.state, self.control
                        )

                        # get leaning rate before update
                        learning_rate = self._get_learning_rate()

                        if not self.accelerator.optimizer_step_was_skipped:
                            # Delay optimizer scheduling until metrics are generated
                            if not isinstance(
                                self.lr_scheduler,
                                torch.optim.lr_scheduler.ReduceLROnPlateau,
                            ):
                                self.lr_scheduler.step()

                        model.zero_grad()
                        self.state.global_step += 1
                        self.state.epoch = (
                            epoch + (step + 1 + steps_skipped) / steps_in_epoch
                        )
                        self.control = self.callback_handler.on_step_end(
                            args, self.state, self.control
                        )
                        self._maybe_log_save_evaluate(
                            tr_loss,
                            grad_norm,
                            model,
                            trial,
                            epoch,
                            ignore_keys_for_eval,
                            start_time,
                            learning_rate=learning_rate,
                        )
                    else:
                        self.control = self.callback_handler.on_substep_end(
                            args, self.state, self.control
                        )

                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if (
                        self.control.should_epoch_stop
                        or self.control.should_training_stop
                    ):
                        if is_torch_xla_available():
                            xm.mark_step()
                        break
                # We also need to break out of the nested loop
                if self.control.should_epoch_stop or self.control.should_training_stop:
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems not to be a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(
                args, self.state, self.control
            )
            self._maybe_log_save_evaluate(
                tr_loss,
                grad_norm,
                model,
                trial,
                epoch,
                ignore_keys_for_eval,
                start_time,
                learning_rate=learning_rate,
            )

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break
            epoch += 1

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info(
            "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n"
        )
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(
            self.state.global_step, 0.001
        )  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(
            use_mtime=False, output_dir=run_dir
        )

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if (
            self.args.should_save
            and self.state.best_model_checkpoint is not None
            and self.args.save_total_limit == 1
        ):
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(
                        f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit"
                    )
                    shutil.rmtree(checkpoint, ignore_errors=True)

        self.control = self.callback_handler.on_train_end(
            args, self.state, self.control
        )

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

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

            logger.info("Train exited ... waiting for commands")

            # Check if we have a load_checkpoint command
            if self._mixin_control_queue.empty():
                logger.info("Empty wrapper control queue, waiting for commands...")
                break

            is_load = False
            load_config = None
            while not self._mixin_control_queue.empty():
                cmd = self._mixin_control_queue.get()
                if cmd.command == LOAD_CHECKPOINT:
                    self._load_message = cmd
                    is_load = True
                    load_config = json.loads(cmd.args)
                elif cmd.command == STOP_TRAINING:
                    self._load_message = None
                    is_load = False
                    break

            if is_load:
                logger.info(f"Received load_checkpoint command: {self._load_message}")
                logger.info(f"load config: {load_config}")

                ckpt_info = self._server.get_checkpoint_info(load_config["uuid"])
                if ckpt_info is None:
                    logger.warning(f"Checkpoint with UUID {load_config['uuid']} not found.")
                    break

                kwargs["resume_from_checkpoint"] = ckpt_info["checkpoint_dir"]
                new_branch_info = self._server.fork_branch(
                    ckpt_info["branch_id"], load_config["branch_name"]
                )
                load_config["branch_info"] = new_branch_info
                self._server.enqueue_event(
                    {
                        "status": "success",
                        "command": LOAD_CHECKPOINT,
                        "args": json.dumps(load_config),
                        "uuid": self._load_message.uuid,
                        "time": self._load_message.time,
                    },
                )
                self._load_message = None
            else:
                break

        self._server.stop()
