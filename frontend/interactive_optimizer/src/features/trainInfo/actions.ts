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
          const trainInfo: TrainInfoData = {
            startTime: response.data.start_time,
            status: response.data.status,
            runName: response.data.run_name || "",
          } as TrainInfoData;
          dispatch(updateTrainInfo(trainInfo));
          if (response.data.status === "running") {
            return true;
          }
        }
      } catch (error) {
        console.error("Error fetching training state:", error);
      } finally {
        attempt++;
        await sleep(1000); // Wait 1 second before retrying
        console.log(`Attempt ${attempt + 1} of ${maxAttempts}`);
      }
    }
    return rejectWithValue("Training state not ready after multiple attempts");
  }
);
