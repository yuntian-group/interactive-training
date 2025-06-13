import type { Middleware } from "@reduxjs/toolkit";
import { WebSocketActionTypes, type WebSocketActions } from "./types";
import { createWebSocket } from "../../api/api";
import { updateCommandStatus } from "../trainCommand/reducer";
import type { TrainCommandData } from "../trainCommand/types";

let socket: WebSocket | null = null;

export const websocketMiddleware: Middleware =
  (store) => (next) => (action: unknown) => {
    const wsAction = action as WebSocketActions;
    switch (wsAction.type) {
      case WebSocketActionTypes.CONNECT:
        socket = createWebSocket(wsAction.payload.url);
        socket.onmessage = (event) => {
          // store.dispatch({
          //   type: WebSocketActionTypes.RECEIVED,
          //   payload: JSON.parse(event.data),
          // });

          const msg = JSON.parse(event.data);

          const msgTrainCommandData: TrainCommandData = {
            uuid: msg.uuid,
            command: msg.command,
            args: msg.args,
            time: msg.time,
            status: msg.status || "received",
          };

          store.dispatch(updateCommandStatus(msgTrainCommandData));
          console.log("Received WebSocket message:", msg);
        };
        socket.onclose = () => {
          store.dispatch({ type: WebSocketActionTypes.CLOSED });
        };
        break;

      case WebSocketActionTypes.SEND:
        if (socket && socket.readyState === WebSocket.OPEN) {
          socket.send(JSON.stringify(wsAction.payload));
        }
        break;

      case WebSocketActionTypes.DISCONNECT:
        if (socket) socket.close();
        socket = null;
        break;
      default:
        break;
    }
    return next(action);
  };
