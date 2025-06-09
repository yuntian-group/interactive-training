import { createAsyncThunk } from "@reduxjs/toolkit";
import { getTrainingState } from "../../api/api";

export const trainCommand = createAsyncThunk(
  "trainInfo/getTrainState",
  async (_, { rejectWithValue }) => {
    try {
      const response = await getTrainingState();
      return response;
    } catch (error) {
      console.error("Error fetching training state:", error);
      return rejectWithValue("Failed to fetch training state");
    }
  }
);
