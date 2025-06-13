import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import { getTrainInfoForInitializaiton } from "./actions";
import type trainInfoState from "./types";
import type { TrainInfoData } from "./types";

const initialState: trainInfoState = {
  trainInfo: {
    start_time: 0,
    status: "unknown",
  },
  loading: false,
  error: null,
};

const trainInfoSlice = createSlice({
  name: "trainInfo",
  initialState,
  reducers: {
    updateTrainInfo: (state, action: PayloadAction<TrainInfoData>) => {
      state.trainInfo = action.payload;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(getTrainInfoForInitializaiton.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(getTrainInfoForInitializaiton.fulfilled, (state) => {
        state.loading = false;
        state.error = null;
      })
      .addCase(getTrainInfoForInitializaiton.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error?.message || "Failed to execute command";
      });
  },
});

export const { updateTrainInfo } = trainInfoSlice.actions;
export default trainInfoSlice.reducer;
