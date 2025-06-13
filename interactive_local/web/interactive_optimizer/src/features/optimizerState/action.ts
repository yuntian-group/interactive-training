import { createAsyncThunk } from "@reduxjs/toolkit";
import { getOptimizerState } from "../../api/api";

export const getOptimizerStateFromServer = createAsyncThunk(
  "optimizerState/getOptimizerStateFromServer",
  async (_, { rejectWithValue }) => {
    try {
      const response = await getOptimizerState();
      return response;
    } catch (error) {
      console.error("Error fetching training state:", error);
      return rejectWithValue("Failed to fetch training state");
    }
  }
);
