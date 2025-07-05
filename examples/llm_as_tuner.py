import os
import json
import uuid
import time
import random
import openai
import requests
import websocket

HTTP_HOST = "http://localhost:9876"
INSTRUCTION_ENDPOINT = "/api/command/"
WEBSOCKET_HOST = "ws://localhost:9876/ws/message/"

PAUSE_REQUEST = {
    "command": "pause_training",
    "args": "",
    "uuid": str(uuid.uuid4()),
    "time": time.time(),
    "status": "requested",
}

RESUME_REQUEST = {
    "command": "resume_training",
    "args": "",
    "uuid": str(uuid.uuid4()),
    "time": time.time(),
    "status": "requested",
}

UPDATE_LR_REQUEST = {
    "command": "update_optimizer",
    "args": "",
    "uuid": str(uuid.uuid4()),
    "time": time.time(),
    "status": "requested",
}

PROMPT_TEMPLATE = open("examples/llm_prompt_template.md").read()

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


train_loss = []
train_steps = []
lr_history = []

eval_loss = []
eval_steps = []


def build_log_history_for_prompt(cur_step: int):
    # Build prompt for LLM using recent history, downsampled to limit tokens
    max_points = 20
    # current learning rate
    cur_lr = lr_history[-1] if lr_history else ""

    # downsample function
    # helper to compute sample indices biased towards recent
    def get_indices(n):
        if n <= max_points:
            return list(range(n))
        exponent = 2.0  # power >1 biases toward recent
        return [
            int(((i / (max_points - 1)) ** exponent) * (n - 1))
            for i in range(max_points)
        ]

    # get sampled histories with steps
    lr_pairs = [
        (train_steps[idx], lr_history[idx]) for idx in get_indices(len(lr_history))
    ]
    train_pairs = [
        (train_steps[idx], train_loss[idx]) for idx in get_indices(len(train_loss))
    ]
    eval_pairs = [
        (eval_steps[idx], eval_loss[idx]) for idx in get_indices(len(eval_loss))
    ]
    # format history strings showing step:value
    lr_history_str = ", ".join(f"{step}:{val:.6f}" for step, val in lr_pairs)
    train_loss_history_str = ", ".join(f"{step}:{val:.4f}" for step, val in train_pairs)
    eval_loss_history_str = ", ".join(f"{step}:{val:.4f}" for step, val in eval_pairs)
    # fill template
    prompt = (
        PROMPT_TEMPLATE.replace("{{current_step}}", str(cur_step))
        .replace("{{current_lr}}", str(cur_lr))
        .replace("{{lr_history}}", lr_history_str)
        .replace("{{train_loss_history}}", train_loss_history_str)
        .replace("{{valid_loss_history}}", eval_loss_history_str)
    )
    return prompt


def parse_action(llm_response: str):
    print("LLM response:", llm_response)
    try:
        action = json.loads(llm_response)["action"]
        return action
    except json.JSONDecodeError:
        print("Failed to parse LLM response as JSON.")
        return None


def call_llm_agent_for_lr_update(cur_step: int):
    print("Calling LLM agent for learning rate update...")
    p = build_log_history_for_prompt(cur_step)
    resp = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {
                "role": "user",
                "content": p,
            },
        ],
        response_format={"type": "json_object"},
    )

    ret = resp.choices[0].message.content

    action = parse_action(ret)
    cur_lr = lr_history[-1]
    if action is None:
        print("No valid action returned by LLM, using default learning rate.")
        return cur_lr
    else:
        if action == "Double":
            return cur_lr * 2
        elif action == "Half":
            return cur_lr / 2
        elif action == "Same":
            return cur_lr
        else:
            print("Unknown action returned by LLM, using random learning rate.")
            return cur_lr


def pause_training():
    print("Pausing training...")
    requests.post(HTTP_HOST + INSTRUCTION_ENDPOINT, json=PAUSE_REQUEST)


def resume_training():
    print("Resuming training...")
    requests.post(HTTP_HOST + INSTRUCTION_ENDPOINT, json=RESUME_REQUEST)


def update_learning_rate(new_lr):
    print(f"Updating learning rate to {new_lr}...")
    UPDATE_LR_REQUEST["args"] = json.dumps(
        {
            "lr": {
                "value": new_lr,
                "name": "lr",
            }
        }
    )
    requests.post(HTTP_HOST + INSTRUCTION_ENDPOINT, json=UPDATE_LR_REQUEST)


def on_message(ws, message):
    msg = json.loads(message)
    if msg["command"] == "log_update":
        log_detail = json.loads(msg["args"])
        print("Log detail:", log_detail)
        if "metrics" in log_detail:
            m = log_detail["metrics"]
            if "eval_loss" in m and m["epoch"] != 0.0:
                eval_loss.append(m["eval_loss"])
                eval_steps.append(m["global_step"])
                cur_step = int(m["global_step"]) + 1
                pause_training()
                new_lr = call_llm_agent_for_lr_update(cur_step)
                resume_training()
                update_learning_rate(new_lr)
            if "loss" in m:
                train_loss.append(m["loss"])
                train_steps.append(m["global_step"])
                lr_history.append(m["learning_rate"])


if __name__ == "__main__":
    ws = websocket.WebSocketApp(WEBSOCKET_HOST, on_message=on_message)
    ws.run_forever(reconnect=True, ping_interval=30, ping_timeout=10)
