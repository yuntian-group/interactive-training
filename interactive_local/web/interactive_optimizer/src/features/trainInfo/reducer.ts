import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import { trainCommand } from "./actions";
import type { trainInfoState } from "./types";

const initialState: trainInfoState = {
  trainState: "",
  loading: false,
  error: null,
  response: null,
};

const trainInfoSlice = createSlice({
  name: "trainInfo",
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(trainCommand.pending, (state) => {
        state.loading = true;
        state.error = null;
        state.response = null;
      })
      .addCase(trainCommand.fulfilled, (state, action: PayloadAction<any>) => {
        state.loading = false;
        state.response = action.payload;
      })
      .addCase(trainCommand.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error?.message || "Failed to execute command";
      });
  },
});

export default trainInfoSlice.reducer;
