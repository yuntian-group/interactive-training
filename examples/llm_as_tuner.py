import openai
import requests
import websocket

API_KEY = ""
HTTP_HOST = "http://localhost:9876"
WEBSOCKET_HOST = "ws://localhost:9876/ws/message/"


def on_message(ws, message):
    print("Received message:", message)


def on_error(ws, error):
    print("Error:", error)


def on_close(ws):
    print("WebSocket closed")


if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        WEBSOCKET_HOST,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    ws.run_forever(reconnect=True, ping_interval=30, ping_timeout=10)
