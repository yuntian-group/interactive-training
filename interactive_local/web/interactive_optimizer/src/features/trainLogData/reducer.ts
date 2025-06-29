import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import { getTrainLogDataFromServer } from "./action";
import type { BranchInfo } from "./type";
import type TrainLogData from "./type";

const MAIN_BRANCH = "main";

const initialState: TrainLogData = {
  // Initial state for train log data
  localDataVersion: 0, // Version of the data structure
  currentBranch: MAIN_BRANCH, // Current branch name
  branchInfo: {}, // Information about each branch
  branchTree: { main: [] }, // Tree structure of branches
  displayBranch: [], // Branch to display in the UI
};

const trainLogDataSlice = createSlice({
  name: "trainLogData",
  initialState,
  reducers: {
    fork: (state, action: PayloadAction<BranchInfo>) => {
      const branchInfo = action.payload;
      const branchId = branchInfo.id;
      // Add the new branch to the branch info
      state.branchInfo[branchId] = branchInfo;
      // Add the new branch to the branch tree
      if (!state.branchTree[branchId]) {
        state.branchTree[branchId] = [];
      }
      state.branchTree[state.currentBranch].push(branchId);
      state.currentBranch = branchId; // Switch to the new branch
      state.localDataVersion += 1; // Increment the local data version
    },
    bumpLocalDataVersion: (state) => {
      console.log("Bumping local data version");
      state.localDataVersion += 1; // Increment the local data version
    },
    resetTrainLogData: () => initialState,
  },
  extraReducers: (builder) => {
    builder.addCase(
      getTrainLogDataFromServer.fulfilled,
      (state, action: PayloadAction<TrainLogData>) => {
        const data = action.payload;
        console.log("Received train log data:", data);
        state.branchTree = data.branchTree;
        state.branchInfo = data.branchInfo;
        state.currentBranch = data.currentBranch;
        state.displayBranch = data.displayBranch;
        state.localDataVersion += 1; // Increment the data version
      }
    );
    builder.addCase(getTrainLogDataFromServer.rejected, (state, action) => {
      console.error("Failed to fetch train log data:", action.payload);
      // TODO
      // Optionally, you can reset the state or handle the error as needed
    });
    builder.addCase(getTrainLogDataFromServer.pending, () => {});
  },
});

export const { resetTrainLogData } = trainLogDataSlice.actions;

export default trainLogDataSlice.reducer;
