import type { Middleware } from "@reduxjs/toolkit";
import { WebSocketActionTypes, type WebSocketActions } from "./types";
import { createWebSocket } from "../../api/api";
import { updateCommandStatus } from "../trainCommand/reducer";
import { updateTrainLogData } from "../trainLogData/reducer";
import type { TrainCommandData } from "../trainCommand/types";
import { getCheckpointStateFromServer } from "../checkpointState/action";

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

          if (msg.command === "log_update") {
            const logUpdateData = JSON.parse(msg.args) as Record<
              string,
              number
            >;
            store.dispatch(updateTrainLogData(logUpdateData));
          } else {
            store.dispatch(updateCommandStatus(msgTrainCommandData));

            if (msg.status === "success") {
              switch (msg.command) {
                case "checkpoint_info_update":
                  store.dispatch(getCheckpointStateFromServer() as any);
              }
            }
          }

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
