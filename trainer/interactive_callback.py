import os
import json
import time
import queue
import httpx
import threading
import websocket
from transformers import TrainerCallback


WS_HOST_TEMPLATE = "ws://{}:{}/ws/{}"
HTTP_HOST_TEMPLATE = "http://{}:{}/"


class InteractiveTrainerCallback(TrainerCallback):
    def __init__(
        self,
        host: str,
        port: int,
        timeout: int = 10,
        model=None,
        optimizer=None,
        lr_scheduler=None,
        train_dataloader=None,
        eval_dataloader=None,
    ):
        super().__init__()
        self._host = host
        self._port = port
        self._http_host = HTTP_HOST_TEMPLATE.format(self._host, self._port)
        self._ws_host = None
        self._run_id = None
        self._timeout = timeout
        self._model = model
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._train_dataloader = train_dataloader
        self._eval_dataloader = eval_dataloader
        self._websocket = None
        self._instruction_queue = queue.Queue()
        self._response_queue = queue.Queue()
        self._running = False
        self._listen_thread = None
        self._response_thread = None

    def _on_message(self, ws, message):
        """Handle incoming messages from the WebSocket."""
        print(f"Received message: {message}")
        self._instruction_queue.put(message)

    def _on_error(self, ws, error):
        """Handle errors from the WebSocket."""
        print(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket closure."""
        print(
            f"WebSocket closed with code {close_status_code} and message: {close_msg}"
        )
        self._websocket = None

    def _listen_for_instructions(self):
        """Listen for instructions from the WebSocket."""
        while self._running:
            try:
                self._websocket = websocket.WebSocketApp(
                    self._ws_host,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                self._websocket.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                print("WebSocket Error. Reconnecting...")
                print(f"Error: {e}")

            if self._running:
                time.sleep(5)  # Wait before trying to reconnect
                print("Attempting to reconnect...")
            else:
                break

    def _respond_to_instructions(self):
        """Respond to instructions from the WebSocket."""
        while self._running:
            try:
                response = self._response_queue.get()
                if self._websocket:
                    self._websocket.send(response)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Response Error: {e}")
                break

    def _queue_response(self, response):
        """Queue a response to be sent back through the WebSocket."""
        try:
            self._response_queue.put(response)
        except queue.Full:
            print("Response queue is full. Dropping response.")
        except Exception as e:
            print(f"Error queuing response: {e}")

    def setup(self):
        if self._running:
            print("Callback is already running.")
            return
        self._running = True

        resp = httpx.post(
            self._http_host + "api/v1/runs/new", json={"test": "hello_world"}
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to create a new run: {resp.text}")

        run_id = resp.json().get("run_id", None)
        if not run_id:
            raise RuntimeError("No run_id returned from the server.")

        print(f"Run ID: {run_id}")

        self._run_id = run_id
        self._ws_host = WS_HOST_TEMPLATE.format(self._host, self._port, run_id)

        print(f"Connecting to WebSocket at {self._ws_host}")

        self._listen_thread = threading.Thread(
            target=self._listen_for_instructions, daemon=True
        )
        self._response_thread = threading.Thread(
            target=self._respond_to_instructions, daemon=True
        )
        self._listen_thread.start()
        self._response_thread.start()

    def cleanup(self):
        """Clean up resources."""
        self._running = False
        if self._websocket:
            self._websocket.close()
            self._websocket = None

        if self._response_thread and self._response_thread.is_alive():
            self._response_queue.put(None)  # sentinel to unblock .get()
            self._response_thread.join(timeout=self._timeout)
        if self._listen_thread:
            self._listen_thread.join(timeout=self._timeout)

        resp = httpx.post(
            self._http_host + "api/v1/runs/{}/stop".format(self._run_id),
            json={"test": "hello_world"},
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Failed to stop the run: {resp.text}")
        print("Callback cleanup complete.")

    def handle_instruction(self, instruction):
        """Handle an instruction received from the WebSocket."""
        print(f"Handle instruction: {instruction}")
        inst_json = json.loads(instruction)

        if "lr" in inst_json:
            if self._optimizer is None:
                inst_json["status"] = "error"
                inst_json["message"] = "Optimizer not set."
            else:
                new_lr = inst_json["lr"]
                for param_group in self._optimizer.param_groups:
                    param_group["lr"] = new_lr
                print(f"Updated learning rate to {new_lr}")
                inst_json["status"] = "success"

        return json.dumps(inst_json)

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.setup()

    def on_step_end(self, args, state, control, **kwargs):
        processed = False
        while True:
            try:
                instruction = self._instruction_queue.get_nowait()
            except queue.Empty:
                break

            processed = True
            ret = self.handle_instruction(instruction)
            self._queue_response(ret)

        if processed:
            print("Processed queued instructions.")

    def on_train_end(self, args, state, control, **kwargs):
        self.cleanup()
