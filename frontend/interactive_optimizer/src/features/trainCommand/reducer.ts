import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import { postTrainCommand } from "./actions";
import type trainCommandState from "./types";
import type { TrainCommandData } from "./types";

const initialState: trainCommandState = {
  loading: false,
  error: null,
  response: null,
  commandRecord: {},
};

const trainCommandSlice = createSlice({
  name: "trainCommand",
  initialState,
  reducers: {
    updateCommandStatus: (state, action: PayloadAction<TrainCommandData>) => {
      // You can also handle args if needed, e.g., store them in state
      const trainCommand = action.payload;
      state.commandRecord[trainCommand.uuid] = {
        ...trainCommand,
        status: trainCommand.status || "pending", // Default to pending if status is not provided
        time: trainCommand.time || Date.now(), // Default to current time if not provided
        command: trainCommand.command || "",
        args: trainCommand.args || "",
      };
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(postTrainCommand.pending, (state) => {
        state.loading = true;
        state.error = null;
        state.response = null;
      })
      .addCase(postTrainCommand.fulfilled, (state, action) => {
        state.loading = false;
        state.response = action.payload;
        console.log(state.response);
      })
      .addCase(postTrainCommand.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error?.message || "Failed to execute command";
      });
  },
});

export const { updateCommandStatus } = trainCommandSlice.actions;
export default trainCommandSlice.reducer;
