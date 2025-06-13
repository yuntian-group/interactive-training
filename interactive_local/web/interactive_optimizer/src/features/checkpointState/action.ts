import { createAsyncThunk } from "@reduxjs/toolkit";
import { getCheckpointInfo } from "../../api/api";

export const getCheckpointStateFromServer = createAsyncThunk(
  "checkpointState/getCheckpointStateFromServer",
  async (_, { rejectWithValue }) => {
    try {
      const response = await getCheckpointInfo();
      return response;
    } catch (error) {
      console.error("Error fetching training state:", error);
      return rejectWithValue("Failed to fetch training state");
    }
  }
);
