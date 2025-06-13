import { createSlice, type PayloadAction } from "@reduxjs/toolkit";

import { getCheckpointStateFromServer } from "./action";

import type checkpointState from "./type";
import type { CheckpointData } from "./type";

const initialState: checkpointState = {
  loading: false,
  error: null,
  response: null,
  state: [],
};

const checkpointStateSlice = createSlice({
  name: "checkpointState",
  initialState,
  reducers: {
    updateCheckpointReducer: (
      state,
      action: PayloadAction<{ type: string; value: CheckpointData[] }>
    ) => {
      const { type, value } = action.payload;
      if (type === "update") {
        state.state = value;
      } else if (type === "reset") {
        state.state = [];
      }
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(getCheckpointStateFromServer.pending, (state) => {
        state.loading = true;
        state.error = null;
        state.response = null;
      })
      .addCase(
        getCheckpointStateFromServer.fulfilled,
        (state, action: PayloadAction<CheckpointData[]>) => {
          state.loading = false;
          state.state = action.payload;
          state.response = action.payload;
        }
      )
      .addCase(getCheckpointStateFromServer.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error?.message || "Failed to fetch checkpoints";
      });
  },
});

export const { updateCheckpointReducer } = checkpointStateSlice.actions;
export default checkpointStateSlice.reducer;
