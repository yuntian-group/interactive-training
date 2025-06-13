import axios from "axios";

const api = axios.create({
  baseURL: "http://127.0.0.1:9876",
  timeout: 10000,
  headers: {
    "Access-Control-Allow-Origin": "*",
    "Content-Type": "application/json",
  },
});

export const getTrainingState = async () => {
  try {
    const response = await api.get("/api/get_info/");
    return response;
  } catch (error) {
    throw error;
  }
};

export const getOptimizerState = async () => {
  try {
    const response = await api.get("/api/get_optimizer_info/");
    return response;
  } catch (error) {
    throw error;
  }
};

export const getCheckpointInfo = async () => {
  try {
    const response = await api.get("/api/get_checkpoint/");
    return response;
  } catch (error) {
    throw error;
  }
};

export const postCommand = async (command: string) => {
  try {
    const response = await api.post("/api/command/", { command });
    return response;
  } catch (error) {
    throw error;
  }
};

export function createWebSocket(url: string): WebSocket {
  return new WebSocket(url);
}
