import json
import queue
import asyncio
import threading
from typing import Dict, List
from pydantic import BaseModel
from uvicorn import Config, Server
from fastapi.middleware.cors import CORSMiddleware
from transformers import Trainer, TrainingArguments
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from trainer.constants import (
    COMMAND_TO_TYPE,
    WRAPER_CONTROL_COMMAND_TYPE,
    SAVE_CHECKPOINT,
    UPDATE_OPTIMIZER,
    CHECKPOINT_INFO_UPDATE,
    TRAIN_STATE_UPDATE,
    LOAD_CHECKPOINT,
    SUCCESS,
    Cmd,
)

from trainer.callbacks import InteractiveCallback, CheckpointCallback, LoggingCallback


class InteractiveServerState:
    checkpoints: List[dict] = []
    commands_dict: Dict[str, Cmd] = {}
    model_infos: Dict[str, str] = {}
    optimizer_states: Dict[str, float] = {}
    start_time: float = 0.0
    status: str = "init"


class TrainLogs:
    local_step: int = 0
    log_values = {}


class InteractiveServer:
    def __init__(
        self,
        host: str,
        port: int,
        output: str = "output",
        timeout: int = 10,
    ):
        self.app = FastAPI()

        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self.host = host
        self.port = port
        self.timeout = timeout

        # Use thread-safe queues for inter-thread communication
        self.events_queue: "queue.Queue[dict]" = queue.Queue()

        all_command_types = set(COMMAND_TO_TYPE.values())
        self.messages_queue_by_type: Dict["queue.Queue[Cmd]"] = {
            k: queue.Queue() for k in all_command_types
        }

        self._train_state = InteractiveServerState()
        self._logs = TrainLogs()
        self._train_state_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._event_listeners: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._app_thread: threading.Thread | None = None
        self.server: Server | None = None
        self.running = False

        self._setup_routes()
        # Register startup/shutdown handlers
        self.app.add_event_handler("startup", self._on_startup)
        self.app.add_event_handler("shutdown", self._on_shutdown)

    def _on_startup(self):
        """
        Called when Uvicorn starts. Capture the running loop so we can
        schedule tasks (e.g., closing websockets) later.
        """
        self._loop = asyncio.get_running_loop()

    def _on_shutdown(self):
        """
        Called when Uvicorn is shutting down. We push sentinel None
        into the events_queue to unblock any waiting WebSocket handlers.
        Then schedule a coroutine to close all open WebSockets.
        """
        # Push one sentinel per active listener so each handler unblocks
        if self._loop:
            # We marshal this into a coroutine on the event loop
            async def _shutdown_coroutine():
                # 1) Enqueue enough None sentinels for each listener
                for _ in list(self._event_listeners):
                    self.events_queue.put(None)

                # 2) Close each WebSocket (fire-and-forget)
                to_close = list(self._event_listeners)
                for ws in to_close:
                    try:
                        await ws.close(code=1000)
                    except Exception:
                        pass
                    self._event_listeners.discard(ws)

            self._loop.call_soon_threadsafe(asyncio.create_task, _shutdown_coroutine())

    def _start_server(self):
        """
        Synchronous entry point that actually runs Uvicorn.
        """
        config = Config(
            app=self.app,
            host=self.host,
            port=self.port,
            timeout_keep_alive=self.timeout,
            loop="asyncio",
        )
        self.server = Server(config=config)
        # This will block until shutdown is triggered
        self.server.run()

    def enqueue_event(self, event: dict):
        """
        Allow external threads (e.g., the training thread) to push events
        that should be broadcast to all connected WebSocket clients.
        """
        self.events_queue.put(event)

    def enqueue_message_by_type(self, command_type: str, cmd: Cmd):
        self.messages_queue_by_type[command_type].put(cmd)

    def update_log_state(self, log: dict):
        with self._log_lock:
            if not isinstance(log, dict):
                print(f"Invalid log format: {log}")
                return
            if "global_step" not in log:
                print(f"Log does not contain 'global_step': {log}")
            self._logs.local_step += 1
            for k, v in log.items():
                if k not in self._logs.log_values:
                    self._logs.log_values[k] = []
                self._logs.log_values[k].append(v)

    def update_server_state(self, event: dict):

        def _update_command_history(event: dict):
            cur_uuid = event.get("uuid", None)
            if cur_uuid is None:
                print("No UUID found in event, cannot update command history.")
                return
            self._train_state.commands_dict[cur_uuid] = Cmd(**event)

        with self._train_state_lock:
            _update_command_history(event)

            if event["status"] != SUCCESS:
                print(f"Received event with error status: {event}")
                return

            if event["command"] not in {
                CHECKPOINT_INFO_UPDATE,
                TRAIN_STATE_UPDATE,
                UPDATE_OPTIMIZER,
            }:
                print("No need to update server")
                return

            if event["command"] == CHECKPOINT_INFO_UPDATE:
                checkpoint_info_json = event.get("args", None)
                if checkpoint_info_json is None:
                    print("No checkpoint info provided in event args.")
                    return
                try:
                    checkpoint_info = json.loads(checkpoint_info_json)
                except json.JSONDecodeError:
                    print(f"Invalid JSON format in event args: {checkpoint_info_json}")
                    return

                self._train_state.checkpoints.append(checkpoint_info)

            elif event["command"] == TRAIN_STATE_UPDATE:

                train_state = json.loads(event.get("args", "{}"))

                print("update train state", train_state)

                self._train_state.model_infos = train_state.get("model_infos", {})
                self._train_state.optimizer_states = train_state.get(
                    "optimizer_states", {}
                )
                self._train_state.start_time = train_state.get("start_time", 0.0)
                self._train_state.status = "running"

            elif event["command"] == UPDATE_OPTIMIZER:
                optimizer_info = json.loads(event.get("args", "{}"))
                self._train_state.optimizer_states.update(optimizer_info)

    def _setup_routes(self):

        @self.app.get("/api/get_info/")
        async def train_state():
            """
            HTTP GET endpoint to retrieve the current training state.
            Returns a JSON representation of the InteractiveServerState.
            """
            with self._train_state_lock:
                return {
                    "start_time": self._train_state.start_time,
                    "status": self._train_state.status,
                }

        @self.app.get("/api/get_optimizer_info/")
        async def get_optimizer_info():
            """
            HTTP GET endpoint to retrieve the current training state.
            Returns a JSON representation of the InteractiveServerState.
            """
            with self._train_state_lock:
                return self._train_state.optimizer_states

        @self.app.get("/api/get_model_info/")
        async def get_model_info():
            """
            HTTP GET endpoint to retrieve the current model information.
            Returns a JSON representation of the InteractiveServerState.
            """
            with self._train_state_lock:
                return self._train_state.model_infos

        @self.app.get("/api/get_checkpoints/")
        async def get_checkpoints():
            """
            HTTP GET endpoint to retrieve the list of checkpoints.
            Returns a JSON representation of the InteractiveServerState.
            """
            with self._train_state_lock:
                return self._train_state.checkpoints

        @self.app.get("/api/get_logs/")
        async def get_logs():
            """
            HTTP GET endpoint to retrieve the training logs.
            """
            with self._log_lock:
                ret = self._logs.log_values.copy()
                return {
                    "local_step": self._logs.local_step,
                    "log_values": ret
                }

        @self.app.post("/api/command/")
        async def receive_command(cmd: Cmd):
            """
            HTTP POST endpoint to receive a command from a client.
            Puts the parsed `Cmd` object into the thread-safe messages_queue.
            """
            # Synchronous put is fine because queue.Queue is thread-safe
            cmd_json = cmd.json()
            print(f"Received command over HTTP: {cmd_json}")

            command_type = COMMAND_TO_TYPE.get(cmd.command, "unknown")
            if command_type not in self.messages_queue_by_type:
                return {"status": "error", "message": "Unknown command type"}

            # broadcast a response back to all WebSocket clients
            response_event = cmd.model_dump()
            response_event["status"] = "pending"
            self.enqueue_event(response_event)

            # Enqueue the command into the appropriate queue
            self.enqueue_message_by_type(command_type, cmd)

            if command_type == LOAD_CHECKPOINT:
                # Special handling for load_checkpoint command
                self.enqueue_message_by_type(WRAPER_CONTROL_COMMAND_TYPE, cmd)

            return {"status": "success"}

        @self.app.websocket("/ws/message/")
        async def websocket_message(websocket: WebSocket):
            """
            WebSocket endpoint that streams events (from events_queue) to clients.
            Uses run_in_executor to block on queue.get() without blocking the event loop.
            """
            await websocket.accept()
            self._event_listeners.add(websocket)
            try:
                loop = asyncio.get_running_loop()
                while True:
                    # Block in a thread for queue.get(); this won't block the event loop.
                    event = await loop.run_in_executor(None, self.events_queue.get)
                    if event is None:
                        # Sentinel received → shutdown this WebSocket
                        break
                    # Broadcast `event` to all connected clients
                    to_remove = set()
                    text = json.dumps(event)
                    print(
                        f"Broadcasting event to {len(self._event_listeners)} clients: {text}"
                    )

                    clients = list(self._event_listeners)
                    results = await asyncio.gather(
                        *(client.send_text(text) for client in clients),
                        return_exceptions=True,
                    )

                    to_remove = set()
                    for client, result in zip(clients, results):
                        if isinstance(result, Exception):
                            print(f"Error sending to {client.client}: {result}")
                            to_remove.add(client)
                    self._event_listeners -= to_remove

            except WebSocketDisconnect:
                print(f"WebSocket disconnected: {websocket.client}")
            finally:
                # Ensure this websocket is no longer in the set
                self._event_listeners.discard(websocket)

    def run(self):
        """
        Start Uvicorn in a separate daemon thread.
        """
        if self.running:
            print("Server already running.")
            return

        if self._app_thread is not None and self._app_thread.is_alive():
            print("Server thread is already alive.")
            return

        self.running = True
        print(f"Starting server at {self.host}:{self.port} (timeout={self.timeout}s).")
        thread = threading.Thread(target=self._start_server, daemon=True)
        self._app_thread = thread
        thread.start()

    def stop(self):
        """
        Signal Uvicorn to shut down and close WebSockets.
        """
        if not self.running:
            return

        self.running = False
        if self.server is not None:
            self.server.should_exit = True

        # The actual shutdown logic (pushing sentinels & closing WebSockets)
        # is handled in the on_shutdown event of FastAPI/Uvicorn.

        # Wait a short while for the thread to finish
        if self._app_thread:
            self._app_thread.join(timeout=5)
            self._app_thread = None


class InteractiveTrainingWrapper:
    def __init__(
        self,
        trainer: Trainer,
        host: str = "127.0.0.1",
        port: int = 9876,
        timeout: int = 10,
    ):
        super().__setattr__("trainer", trainer)
        self.train_args = trainer.args
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

        # TODO: Support other libraries like TensorBoard, MLflow, etc.

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying Trainer instance.
        """
        return getattr(self.trainer, name)

    def __setattr__(self, name, value):
        if name in self.__dict__ or hasattr(self, name) or name == "trainer":
            super().__setattr__(name, value)
        else:
            setattr(self.trainer, name, value)

    def _initialize_callback(self):

        update_callback = InteractiveCallback(
            cmd_queue=self._server.messages_queue_by_type["update"],
            event_queue=self._server.events_queue,
            server_state_update_callback=self._server.update_server_state,
        )
        checkpoint_callback = CheckpointCallback(
            cmd_queue=self._server.messages_queue_by_type["wrapper_control"],
            event_queue=self._server.events_queue,
            server_state_update_callback=self._server.update_server_state,
        )

        logging_callback = LoggingCallback(
            event_queue=self._server.events_queue,
            server_state_update_callback=self._server.update_log_state,
        )
        self.trainer.add_callback(update_callback)
        self.trainer.add_callback(checkpoint_callback)
        self.trainer.add_callback(logging_callback)

    def train(self, **kwargs):
        """
        Start the training process with the Trainer instance.
        This method will run the training loop and listen for commands
        from the InteractiveServer.
        """
        self._server.run()
        self._initialize_callback()

        while True:
            self.trainer.train(**kwargs)

            # Check if we have a load_checkpoint command
            if self._wrapper_control_queue.empty():
                break

            is_load = False
            load_config = None
            while not self._wrapper_control_queue.empty():
                cmd = self._wrapper_control_queue.get()
                if cmd.command == LOAD_CHECKPOINT:
                    self._load_message = cmd
                    print(f"Received load_checkpoint command: {cmd}")
                    is_load = True
                    load_config = cmd.args

            if is_load:
                self._load_message = cmd
                print(f"Received load_checkpoint command: {cmd}")
                print(f"load config: {load_config}")
                pass

        self._server.stop()
