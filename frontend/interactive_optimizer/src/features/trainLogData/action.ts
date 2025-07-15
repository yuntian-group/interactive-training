import { createAsyncThunk } from "@reduxjs/toolkit";
import { getTrainLogData } from "../../api/api";

import type TrainLogData from "./type";
import type { SingleMetricsPoint } from "./type";
import { loadInitialSnapshot } from "./logBuffers";

export const computeDisplayBranch = (
  branchTree: Record<string, string[]>,
  branchInfo: Record<string, { id: string; wall_time: number; parent: string }>,
  currentBranch: string
): string[] => {
  const displayBranch: string[] = [currentBranch];
  const currentBranchParent: string | null =
    branchInfo[currentBranch]?.parent || null;

  let tmpParent = currentBranchParent;

  while (tmpParent && tmpParent !== "main") {
    displayBranch.unshift(tmpParent);
    tmpParent = branchInfo[tmpParent]?.parent || null;
  }

  // Add all siblings of the current branch

  if (!currentBranchParent || !(currentBranchParent in branchTree)) {
    console.warn(
      `Current branch parent "${currentBranchParent}" not found in branch tree.`
    );
  } else {
    const siblings = branchTree[currentBranchParent] || [];

    for (const sibling of siblings) {
      if (sibling !== currentBranch) {
        displayBranch.push(sibling);
      }
    }
  }

  if (currentBranch !== "main") {
    displayBranch.push("main");
  }
  return displayBranch;
};

export const getTrainLogDataFromServer = createAsyncThunk(
  "trainLogData/getTrainLogDataFromServer",
  async (_, { rejectWithValue }) => {
    try {
      const response = await getTrainLogData();
      const data = response.data;

      const trainLogData: TrainLogData = {
        branchTree: data.branch_tree || { main: [] }, // Tree structure of branches
        branchInfo: data.branch_info || {}, // Information about each branch
        currentBranch: data.current_branch || "main", // Current branch name
        displayBranch: [],
        curLog: "", // Current log content
        localDataVersion: 0, // Version of the data structure
        localLogVersion: 0, // Version of the log data
      };

      // Compute the display branch based on the current branch and its parent
      trainLogData.displayBranch = computeDisplayBranch(
        trainLogData.branchTree,
        trainLogData.branchInfo,
        trainLogData.currentBranch
      );

      const initialBranchLogs =
        data.branched_logs || ({} as Record<string, SingleMetricsPoint[]>);
      loadInitialSnapshot(initialBranchLogs);
      return trainLogData;
    } catch (error) {
      console.error("Error executing train command:", error);
      return rejectWithValue("Failed to execute train command");
    }
  }
);
