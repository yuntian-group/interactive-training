import json
import openai
import requests
import websocket

API_KEY = ""
HTTP_HOST = "http://localhost:9876"
INSTRUCTION_ENDPOINT = "/api/instruction/"
WEBSOCKET_HOST = "ws://localhost:9876/ws/message/"


train_loss = []
steps = []
eval_loss = []


def on_message(ws, message):
    print("Received message:", message)
    msg = json.loads(message)
    if msg["command"] == "log_update":
        log_detail = msg["args"]
        if "eval_loss" in log_detail:
            eval_loss.append(float(log_detail["eval_loss"]))
        else:
            train_loss.append(float(log_detail["loss"]))
            steps.append(int(log_detail["step"]))

        print(
            train_loss,
        )


if __name__ == "__main__":
    ws = websocket.WebSocketApp(WEBSOCKET_HOST, on_message=on_message)
    ws.run_forever(reconnect=True, ping_interval=30, ping_timeout=10)
