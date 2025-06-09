import { createAsyncThunk } from "@reduxjs/toolkit";
import { postCommand } from "../../api/api";

export const trainCommand = createAsyncThunk(
  "trainCommand/execute",
  async (command: string, { rejectWithValue }) => {
    try {
      const response = await postCommand(command);
      return response;
    } catch (error) {
      console.error("Error executing train command:", error);
      return rejectWithValue("Failed to execute train command");
    }
  }
);
