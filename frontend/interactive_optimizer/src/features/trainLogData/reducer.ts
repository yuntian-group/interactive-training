import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import { computeDisplayBranch, getTrainLogDataFromServer } from "./action";
import type { BranchInfo } from "./type";
import type TrainLogData from "./type";

const MAIN_BRANCH = "main";
const MOD = 1e9 + 7; // A large prime number for modulo operations

const initialState: TrainLogData = {
  // Initial state for train log data
  localDataVersion: 0, // Version of the data structure
  localLogVersion: 0, // Version of the log data
  curLog: "", // Current log content
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
      const parentBranch = branchInfo.parent || state.currentBranch;

      // Add the new branch to the branch info
      state.branchInfo[branchId] = branchInfo;
      // Add the new branch to the branch tree
      if (!state.branchTree[branchId]) {
        state.branchTree[branchId] = [];
      }
      if (!state.branchTree[parentBranch]) {
        state.branchTree[parentBranch] = [];
      }
      state.branchTree[parentBranch].push(branchId);
      state.currentBranch = branchId; // Switch to the new branch
      state.displayBranch = computeDisplayBranch(
        state.branchTree,
        state.branchInfo,
        branchId
      ); // Update the display branch
      console.log("Updated display branch:", state.displayBranch);
    },
    bumpLocalDataVersion: (state) => {
      state.localDataVersion += 1; // Increment the local data version
      state.localDataVersion %= MOD; // Ensure it stays within bounds
    },
    updateCurrentLog: (state, action: PayloadAction<string>) => {
      state.curLog = action.payload; // Update the current log content
      state.localLogVersion += 1; // Increment the log version when updating the log
      state.localLogVersion %= MOD; // Ensure it stays within bounds
    },
    bumpLogVersion: (state) => {
      state.localLogVersion += 1; // Increment the local log version
      state.localLogVersion %= MOD; // Ensure it stays within bounds
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
    builder.addCase(getTrainLogDataFromServer.rejected, (_state, action) => {
      console.error("Failed to fetch train log data:", action.payload);
      // TODO
      // Optionally, you can reset the state or handle the error as needed
    });
    builder.addCase(getTrainLogDataFromServer.pending, () => {});
  },
});

export const { resetTrainLogData } = trainLogDataSlice.actions;

export default trainLogDataSlice.reducer;
