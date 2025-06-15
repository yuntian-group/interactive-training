export const WebSocketActionTypes = {
  CONNECT: "WS_CONNECT",
  DISCONNECT: "WS_DISCONNECT",
  SEND: "WS_SEND",
  RECEIVED: "WS_RECEIVED",
  CLOSED: "WS_CLOSED",
} as const;

export interface WSConnectAction {
  type: "WS_CONNECT";
  payload: { url: string };
}

export interface WSDisconnectAction {
  type: "WS_DISCONNECT";
}

export interface WSSendAction {
  type: "WS_SEND";
  payload: any;
}

export interface WSReceivedAction {
  type: "WS_RECEIVED";
  payload: any;
}

export interface WSCloseAction {
  type: "WS_CLOSED";
}

export type WebSocketActions =
  | WSConnectAction
  | WSDisconnectAction
  | WSSendAction
  | WSReceivedAction
  | WSCloseAction;

export interface trainEventState {
  connected: boolean;
}
