"""
Minimal run-/worker-/client relay server.

HTTP
----
POST /api/v1/runs/new                → {"run_id": "..."}
POST /api/v1/runs/{run_id}/command   body={"command": "..."}
WS   /ws/{run_id}                    (exactly ONE worker per run)
WS   /ws/client                      (many clients; subscribe cmd below)

A client subscribes by sending:
    {"action": "subscribe", "run_id": "..."}
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Dict, Set

import uvicorn
from fastapi import (
    Body,
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# 1.  State containers
# --------------------------------------------------------------------------- #


@dataclass
class RunState:
    info: dict
    worker_ws: WebSocket | None = None
    to_worker: asyncio.Queue[str] = field(default_factory=asyncio.Queue)
    clients: Set[str] = field(default_factory=set)  # client IDs


@dataclass
class ClientState:
    ws: WebSocket
    queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)


class ServerState:
    """Singleton container for all in-memory state (safe for simple single-process demo)."""

    def __init__(self) -> None:
        self.runs: Dict[str, RunState] = {}
        self.clients: Dict[str, ClientState] = {}

    # Convenience helpers --------------------------------------------------- #
    def get_run(self, run_id: str) -> RunState:
        if run_id not in self.runs:
            raise HTTPException(status_code=404, detail="Invalid run ID")
        return self.runs[run_id]


STATE = ServerState()  # global singleton

# --------------------------------------------------------------------------- #
# 2.  FastAPI setup
# --------------------------------------------------------------------------- #

app = FastAPI(title="Interactive Controller Relay")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- HTTP routes ---------------------------------------------------- #


class NewRunBody(BaseModel):
    run_info: dict = {}


@app.post("/api/v1/runs/new")
async def new_run(body: NewRunBody):
    run_id = str(uuid.uuid4())
    STATE.runs[run_id] = RunState(info=body.run_info)
    return {"run_id": run_id}


@app.post("/api/v1/runs/{run_id}/stop")
async def stop_run(run_id: str):
    run_state = STATE.get_run(run_id)
    if run_state.worker_ws is not None:
        await run_state.worker_ws.close(code=1000, reason="Run stopped")
    # cleanup: remove all clients from this run
    for client_id in run_state.clients:
        await STATE.clients[client_id].queue.put(f"Run {run_id} stopped")
    run_state.clients.clear()
    STATE.runs.pop(run_id, None)
    return {"status": "ok", "run_id": run_id}


class CommandBody(BaseModel):
    command: str


@app.post("/api/v1/runs/{run_id}/command")
async def run_command(run_id: str, body: CommandBody):
    run_state = STATE.get_run(run_id)
    await run_state.to_worker.put(body.command)
    # broadcast a short acknowledgement to every subscribed client
    for client_id in run_state.clients:
        await STATE.clients[client_id].queue.put(f"{body.command} initiated")
    return {"status": "ok", "run_id": run_id}


# ---------- WebSocket: client --------------------------------------------- #


@app.websocket("/ws/client")
async def client_ws(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    STATE.clients[client_id] = ClientState(ws=websocket)
    print(f"[client {client_id}] connected")

    async def send_to_client():
        try:
            while True:
                msg = await STATE.clients[client_id].queue.get()
                await websocket.send_text(msg)
        except WebSocketDisconnect:
            pass

    async def recv_from_client():
        try:
            while True:
                raw = await websocket.receive_text()
                # very small protocol: JSON with {"action": ..., ...}
                # we parse only 'subscribe' for now
                import json

                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if msg.get("action") == "subscribe":
                    run_id = msg.get("run_id")
                    run_state = STATE.get_run(run_id)
                    run_state.clients.add(client_id)
                    await STATE.clients[client_id].queue.put(
                        json.dumps(
                            {"action": "subscribe", "run_id": run_id, "status": 1}
                        )
                    )
        except WebSocketDisconnect:
            pass

    await run_until_first_complete(send_to_client(), recv_from_client())

    # cleanup
    print(f"[client {client_id}] disconnected")
    # remove client from all runs it subscribed to
    for run_state in STATE.runs.values():
        run_state.clients.discard(client_id)
    STATE.clients.pop(client_id, None)


# ---------- WebSocket: worker --------------------------------------------- #


@app.websocket("/ws/{run_id}")
async def worker_ws(websocket: WebSocket, run_id: str):
    run_state = STATE.get_run(run_id)

    # allow exactly one worker per run
    if run_state.worker_ws is not None:
        await websocket.close(code=4409, reason="Worker already connected")
        return

    await websocket.accept()
    run_state.worker_ws = websocket
    print(f"[run {run_id}] worker connected")

    async def send_commands():
        """Pump queued commands to the worker."""
        try:
            while True:
                cmd = await run_state.to_worker.get()
                await websocket.send_text(cmd)
        except WebSocketDisconnect:
            pass

    async def relay_from_worker():
        """Forward every message from worker to all subscribed clients."""
        try:
            while True:
                data = await websocket.receive_text()
                for client_id in run_state.clients:
                    await STATE.clients[client_id].queue.put(data)
        except WebSocketDisconnect:
            pass

    await run_until_first_complete(send_commands(), relay_from_worker())

    # cleanup
    print(f"[run {run_id}] worker disconnected")
    run_state.worker_ws = None


# --------------------------------------------------------------------------- #
# 3.  Utility: run two coroutines until one finishes ------------------------ #
# --------------------------------------------------------------------------- #


async def run_until_first_complete(*aws):
    """Run all tasks; cancel the rest as soon as one finishes."""
    tasks = [asyncio.create_task(x) for x in aws]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    for t in pending:  # drop the other side
        t.cancel()
    for d in done:
        # propagate first exception (if any) to caller
        exc = d.exception()
        if exc:
            raise exc


# --------------------------------------------------------------------------- #
# 4.  Entry point ----------------------------------------------------------- #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=9876, reload=True)
