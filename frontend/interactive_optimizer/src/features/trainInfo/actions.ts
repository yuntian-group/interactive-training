import { createAsyncThunk } from "@reduxjs/toolkit";
import { getTrainingState } from "../../api/api";
import type { TrainInfoData } from "./types";
import { updateTrainInfo } from "./reducer";

const sleep = (ms: number) => new Promise((res) => setTimeout(res, ms));

export const getTrainInfoForInitializaiton = createAsyncThunk(
  "trainInfo/getTrainInfoForInitialization",
  async (_, { dispatch, rejectWithValue }) => {
    const maxAttempts = 30;
    let attempt = 0;

    while (attempt < maxAttempts) {
      try {
        const response = await getTrainingState();
        if (response && response.data) {
          console.log("Received training state:", response.data);
          dispatch(updateTrainInfo(response.data as TrainInfoData));
          console.log("Training state fetched successfully:", response.data);
          if (response.data.status === "running") {
            console.log("Training is running, proceeding with initialization.");
            return true;
          }
        }
      } catch (error) {
        console.log("Error fetching training state:", error);
      } finally {
        attempt++;
        await sleep(1000); // Wait 1 second before retrying
        console.log(`Attempt ${attempt + 1} of ${maxAttempts}`);
      }
    }
    return rejectWithValue("Training state not ready after multiple attempts");
  }
);
