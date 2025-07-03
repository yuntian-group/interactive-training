import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import { getOptimizerStateFromServer } from "./action";
import type optimizerState from "./type";
import type { OptimizerData } from "./type";
import type { TrainCommandData } from "../trainCommand/types";

const initialState: optimizerState = {
  loading: false,
  error: null,
  response: null,
  optimizer_state: {},
};

const optimizerStateSlice = createSlice({
  name: "optimizerState",
  initialState,
  reducers: {
    updateOptimizerReducer: (
      state,
      action: PayloadAction<TrainCommandData>
    ) => {
      const result: TrainCommandData = action.payload;
      if (result.status === "success") {
        const updateDict = JSON.parse(result.args);
        state.optimizer_state = {
          ...state.optimizer_state,
          ...Object.fromEntries(
            Object.entries(updateDict).map(([key, value]) => [
              key,
              {
                name: key,
                value: value as number,
              } as OptimizerData,
            ])
          ),
        };
      } else if (result.status === "failed") {
        // Handle the case where the command failed
        console.error("Command failed:", result);
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(getOptimizerStateFromServer.pending, (state) => {
        state.loading = true;
        state.error = null;
        state.response = null;
      })
      .addCase(
        getOptimizerStateFromServer.fulfilled,
        (state, action: PayloadAction<Record<string, OptimizerData>>) => {
          state.loading = false;
          state.response = action.payload;
          state.optimizer_state = action.payload;
        }
      )
      .addCase(getOptimizerStateFromServer.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error?.message || "Failed to execute command";
      });
  },
});

export const { updateOptimizerReducer } = optimizerStateSlice.actions;
export default optimizerStateSlice.reducer;
