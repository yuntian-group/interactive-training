import { createAsyncThunk } from "@reduxjs/toolkit";
import { getCheckpointInfo } from "../../api/api";
import type { CheckpointData } from "./type";

export const getCheckpointStateFromServer = createAsyncThunk(
  "checkpointState/getCheckpointStateFromServer",
  async (_, { rejectWithValue }) => {
    try {
      const response = await getCheckpointInfo();
      const ckpt_data = response.data;
      if (!ckpt_data || !Array.isArray(ckpt_data)) {
        console.error("Invalid checkpoint data format:", ckpt_data);
        return rejectWithValue("Invalid checkpoint data format");
      }

      const checkpoint_data_list = ckpt_data
        .map((item: any) => {
          if (
            typeof item.time !== "number" ||
            typeof item.uuid !== "string" ||
            typeof item.checkpoint_dir !== "string"
          ) {
            console.error("Invalid checkpoint item format:", item);
            return null; // Skip invalid items
          }
          return {
            time: item.time, // Assuming time is a number
            checkpoint_dir: item.checkpoint_dir, // Assuming path is a string
            uuid: item.uuid, // Assuming uuid is a string
          } as CheckpointData;
        })
        .filter((item: CheckpointData | null) => item !== null); // Filter out any null items
      return checkpoint_data_list;
    } catch (error) {
      console.error("Error fetching training state:", error);
      return rejectWithValue("Failed to fetch training state");
    }
  }
);
