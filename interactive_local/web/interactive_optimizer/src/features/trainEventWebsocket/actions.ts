import { WebSocketActionTypes } from "./types";

export const connectWebSocket = (url: string) => ({
  type: WebSocketActionTypes.CONNECT,
  payload: { url },
});

export const disconnectWebSocket = () => ({
  type: WebSocketActionTypes.DISCONNECT,
});
