import os
import json
import queue
import asyncio
import threading
from typing import Dict, List
from uvicorn import Config, Server
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from trainer.constants import (
    COMMAND_TO_TYPE,
    WRAPER_CONTROL_COMMAND_TYPE,
    UPDATE_OPTIMIZER,
    CHECKPOINT_INFO_UPDATE,
    TRAIN_STATE_UPDATE,
    LOAD_CHECKPOINT,
    SUCCESS,
    Cmd,
)


class InteractiveServerState:
    checkpoints: List[dict] = []
    commands_dict: Dict[str, Cmd] = {}
    model_infos: Dict[str, any] = {}
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

    def get_checkpoint_info(self, uuid: str) -> dict:
        """
        Retrieve checkpoint information by UUID.
        Returns the checkpoint info if found, otherwise None.
        """
        with self._train_state_lock:
            for ckpt in self._train_state.checkpoints:
                if ckpt["uuid"] == uuid:
                    return ckpt
        return None

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

                new_ckpt_info = []
                dedup = set()

                for ckpt_info in self._train_state.checkpoints:
                    if ckpt_info["uuid"] in dedup:
                        continue
                    dedup.add(ckpt_info["uuid"])
                    if os.path.exists(ckpt_info["checkpoint_dir"]):
                        new_ckpt_info.append(ckpt_info)

                new_ckpt_info = list(
                    sorted(new_ckpt_info, key=lambda x: x["global_step"])
                )

                print(new_ckpt_info)

                self._train_state.checkpoints = new_ckpt_info

            elif event["command"] == TRAIN_STATE_UPDATE:

                train_state = json.loads(event.get("args", "{}"))

                # print("update train state", train_state)

                self._train_state.model_infos = train_state.get("model_infos", {})
                self._train_state.optimizer_states = train_state.get(
                    "optimizer_states", {}
                )
                self._train_state.start_time = train_state.get("start_time", 0.0)
                self._train_state.status = "running"

            elif event["command"] == UPDATE_OPTIMIZER:
                optimizer_info = json.loads(event.get("args", "{}"))

                unpacked_update = {
                    k: v["value"]
                    for k, v in optimizer_info.items()
                    if isinstance(v, dict) and "value" in v
                }

                self._train_state.optimizer_states.update(unpacked_update)

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
                return {"local_step": self._logs.local_step, "log_values": ret}

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
                print("Fail to get command to type")
                return {"status": "error", "message": "Unknown command type"}

            # broadcast a response back to all WebSocket clients
            response_event = cmd.model_dump()
            response_event["status"] = "pending"
            self.enqueue_event(response_event)

            if cmd.command == LOAD_CHECKPOINT:
                # Special handling for load_checkpoint command
                print("ADD TO WRAPER_CONTROL_COMMAND_TYPE")
                self.enqueue_message_by_type(WRAPER_CONTROL_COMMAND_TYPE, cmd)

            # Enqueue the command into the appropriate queue
            self.enqueue_message_by_type(command_type, cmd)

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
