import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import { getDatasetInfoFromServer } from "./action";

import type { DatasetInfoState, DatasetInfo } from "./type";

const initialState: DatasetInfoState = {
  datasetInfo: null, // Initially no dataset info is available
  status: "idle", // Initial status of the dataset info fetch
};

const datasetInfoStateSlice = createSlice({
  name: "datasetInfoState",
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(getDatasetInfoFromServer.pending, (state) => {
        state.status = "loading"; // Set status to loading when the request starts
        state.datasetInfo = null; // Reset dataset info to null
        return state;
      })
      .addCase(
        getDatasetInfoFromServer.fulfilled,
        (state, action: PayloadAction<DatasetInfo>) => {
          state.datasetInfo = action.payload; // Update dataset info with the fetched data
          state.status = "succeeded"; // Set status to succeeded if the request is successful
          return state;
        }
      )
      .addCase(getDatasetInfoFromServer.rejected, (state, action) => {
        console.error("Failed to fetch dataset info:", action.error.message);
        state.status = "failed"; // Set status to failed if the request fails
        state.datasetInfo = null; // Reset dataset info to null
        return state;
      });
  },
});

export default datasetInfoStateSlice.reducer;
