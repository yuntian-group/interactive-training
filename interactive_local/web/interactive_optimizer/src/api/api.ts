import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:9876",
  timeout: 10000,
  headers: {
    "Access-Control-Allow-Origin": "*",
    "Content-Type": "application/json",
  },
});

export const getTrainingState = async () => {
  try {
    const response = await api.get("");
    return response.data;
  } catch (error) {
    console.error("Error fetching training state:", error);
    throw error;
  }
};

export const postCommand = async (command: string) => {
  try {
    const response = await api.post("/api/command/", { command });
    return response.data;
  } catch (error) {
    console.error("Error posting command:", error);
    throw error;
  }
};

export function createWebSocket(url: string): WebSocket {
  return new WebSocket(url);
}
