import { createAsyncThunk } from "@reduxjs/toolkit";
import { getDatasetInfo } from "../../api/api";
import type { DatasetInfo } from "./type";

export const getDatasetInfoFromServer = createAsyncThunk(
  "dataset/getDatasetInfoFromServer",
  async (_, { rejectWithValue }) => {
    try {
      const response = await getDatasetInfo();
      const dataset_info = response.data as DatasetInfo; // Assuming the response data is of type DatasetInfo
      console.log("Dataset info fetched successfully:", dataset_info);
      return dataset_info;
    } catch (error) {
      console.error("Error fetching dataset info:", error);
      return rejectWithValue("Failed to fetch dataset info");
    }
  }
);
