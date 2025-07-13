import type { Middleware } from "@reduxjs/toolkit";
import { createWebSocket } from "../../api/api";
import type { BranchInfo } from "../trainLogData/type";
import type { TrainCommandData } from "../trainCommand/types";
import type { SingleMetricsPoint } from "../trainLogData/type";
import { appendNewDataPoint } from "../trainLogData/logBuffers";
import { WebSocketActionTypes, type WebSocketActions } from "./types";
import { getCheckpointStateFromServer } from "../checkpointState/action";
import TermnialHistoryManager from "../terminalHistory/terminalHistoryManager";

let socket: WebSocket | null = null;

const timeDisplayOption: Intl.DateTimeFormatOptions = {
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hour12: false,
};

const logToString = (msg: TrainCommandData): string => {
  //convert timestamp to human-readable format
  // normalize timestamp (if it's in seconds, convert to ms)

  const raw = typeof msg.time === "string" ? parseFloat(msg.time) : msg.time;
  const timestampMs = raw > 1e12 ? raw : raw * 1000;
  const date = new Date(timestampMs);

  const formattedDate = date.toLocaleString("en-US", timeDisplayOption);

  if (msg.command === "log_update") {
    // If the command is log_update, we want to return the log update message
    const metricsStr = JSON.stringify(JSON.parse(msg.args).metrics);
    return `${formattedDate} - ${msg.command}: ${metricsStr}`;
  }

  return `${formattedDate} - ${msg.command} [${
    msg.status ? msg.status : "unknown"
  }]`;
};

const terminalHistoryManager = TermnialHistoryManager.getInstance();

export const websocketMiddleware: Middleware =
  (store) => (next) => (action: unknown) => {
    const wsAction = action as WebSocketActions;
    switch (wsAction.type) {
      case WebSocketActionTypes.CONNECT:
        socket = createWebSocket(wsAction.payload.url);
        socket.onmessage = (event) => {
          const msg = JSON.parse(event.data);

          const msgTrainCommandData: TrainCommandData = {
            uuid: msg.uuid,
            command: msg.command,
            args: msg.args,
            time: msg.time,
            status: msg.status || "received",
          };

          const logStr = logToString(msgTrainCommandData);

          terminalHistoryManager.addToHistory(logStr);

          store.dispatch({
            type: "trainLogData/updateCurrentLog",
            payload: logStr,
          });

          if (msg.command === "log_update") {
            const logUpdateData = JSON.parse(msg.args) as SingleMetricsPoint;
            appendNewDataPoint(logUpdateData);
            store.dispatch({
              type: "trainLogData/bumpLocalDataVersion",
            });
          } else {
            // store.dispatch(updateCommandStatus(msgTrainCommandData));
            if (msg.status === "success") {
              switch (msg.command) {
                case "checkpoint_info_update":
                  store.dispatch(getCheckpointStateFromServer() as any);
                  break;
                case "load_checkpoint":
                  const branchInfo = JSON.parse(msg.args)
                    .branch_info as BranchInfo;
                  store.dispatch({
                    type: "trainLogData/fork",
                    payload: branchInfo,
                  });
                  break;
                case "pause_training":
                  store.dispatch({
                    type: "trainInfo/pauseTrain",
                  });
                  break;
                case "resume_training":
                  store.dispatch({
                    type: "trainInfo/resumeTrain",
                  });
                  break;
                case "stop_training":
                  store.dispatch({
                    type: "trainInfo/stopTrain",
                  });
                  break;
              }
            }
          }

          console.log("Received WebSocket message:", msg);
        };

        socket.onerror = (error) => {
          console.error("WebSocket error occurred:", error);
          store.dispatch({
            type: "trainLogData/updateCurrentLog",
            payload: `${new Date().toLocaleString(
              "en-US",
              timeDisplayOption
            )} - WebSocket Error: Connection error occurred`,
          });
        };

        socket.onclose = (event: CloseEvent) => {
          const { code } = event;
          let closeMessage = `WebSocket connection closed. Code: ${code}`;
          console.log(closeMessage);

          store.dispatch({
            type: "trainLogData/updateCurrentLog",
            payload: `${new Date().toLocaleString(
              "en-US",
              timeDisplayOption
            )} - ${closeMessage}`,
          });

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
