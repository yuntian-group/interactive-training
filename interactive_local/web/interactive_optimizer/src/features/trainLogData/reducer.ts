import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import { getTrainLogDataFromServer } from "./action";
import type TrainLogData from "./type";

const initialState: TrainLogData = {
  steps: [],
  train_log_values: {},
  local_step: 0,
};

const trainLogDataSlice = createSlice({
  name: "trainLogData",
  initialState,
  reducers: {
    updateTrainLogData: (
      state,
      action: PayloadAction<Record<string, number>>
    ) => {
      const log_data = action.payload;
      if (!log_data) {
        return;
      }
      state.local_step += 1;
      if (log_data.global_step !== undefined) {
        state.steps.push(log_data.global_step);
      } else {
        state.steps.push(state.local_step);
      }

      for (const [key, value] of Object.entries(log_data)) {
        if (state.train_log_values.hasOwnProperty(key) === false) {
          state.train_log_values[key] = [];
        }
        state.train_log_values[key].push(value);
      }
      return state;
    },
    resetTrainLogData: () => initialState,
  },
  extraReducers: (builder) => {
    // You can add extra reducers here if needed
    builder.addCase(
      getTrainLogDataFromServer.fulfilled,
      (state, action: PayloadAction<TrainLogData>) => {
        const data = action.payload;
        state.train_log_values = data.train_log_values;
        state.local_step = data.local_step;
        state.steps = data.steps;
      }
    );
    builder.addCase(getTrainLogDataFromServer.rejected, (state, action) => {
      console.error("Failed to fetch train log data:", action.payload);
      // Optionally, you can reset the state or handle the error as needed
      state.steps = [];
      state.train_log_values = {};
      state.local_step = 0;
    });
    builder.addCase(getTrainLogDataFromServer.pending, (state) => {
      // Optionally, you can handle the pending state if needed
      console.log("Fetching train log data...");
    });
  },
});

export const { updateTrainLogData, resetTrainLogData } =
  trainLogDataSlice.actions;

export default trainLogDataSlice.reducer;
