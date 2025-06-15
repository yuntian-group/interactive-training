import { createAsyncThunk } from "@reduxjs/toolkit";
import { getTrainLogData } from "../../api/api";
import type TrainLogData from "./type";

export const getTrainLogDataFromServer = createAsyncThunk(
  "trainLogData/getTrainLogDataFromServer",
  async (_, { rejectWithValue }) => {
    try {
      const response = await getTrainLogData();
      const data = response.data;

      console.log("Received train log data:", data);

      const trainLogData: TrainLogData = {
        steps: data.log_values.global_step,
        train_log_values: data.log_values ? data.log_values : {},
        local_step: data.local_step || 0,
      };
      return trainLogData;
    } catch (error) {
      console.error("Error executing train command:", error);
      return rejectWithValue("Failed to execute train command");
    }
  }
);
