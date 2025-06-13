export enum WebSocketActionTypes {
  CONNECT = "WS_CONNECT",
  DISCONNECT = "WS_DISCONNECT",
  SEND = "WS_SEND",
  RECEIVED = "WS_RECEIVED",
  CLOSED = "WS_CLOSED",
}

export interface WSConnectAction {
  type: WebSocketActionTypes.CONNECT;
  payload: { url: string };
}

export interface WSDisconnectAction {
  type: WebSocketActionTypes.DISCONNECT;
}

export interface WSSendAction {
  type: WebSocketActionTypes.SEND;
  payload: any;
}

export interface WSReceivedAction {
  type: WebSocketActionTypes.RECEIVED;
  payload: any;
}

export interface WSCloseAction {
  type: WebSocketActionTypes.CLOSED;
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
