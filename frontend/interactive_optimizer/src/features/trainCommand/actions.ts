import { createAsyncThunk } from "@reduxjs/toolkit";
import { postCommand } from "../../api/api";
import type { TrainCommandData } from "./types";

export const postTrainCommand = createAsyncThunk(
  "trainCommand/postTrainCommand",
  async (command: TrainCommandData, { rejectWithValue }) => {
    try {
      console.log("Posting train command:", command);
      const response = await postCommand(JSON.stringify(command));
      return response.data;
    } catch (error) {
      console.error("Error executing train command:", error);
      return rejectWithValue("Failed to execute train command");
    }
  }
);
