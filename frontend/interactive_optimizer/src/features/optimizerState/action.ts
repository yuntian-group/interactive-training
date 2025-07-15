import { createAsyncThunk } from "@reduxjs/toolkit";
import { getOptimizerState } from "../../api/api";

export const getOptimizerStateFromServer = createAsyncThunk(
  "optimizerState/getOptimizerStateFromServer",
  async (_, { rejectWithValue }) => {
    try {
      const response = await getOptimizerState();
      const optimizer_data_dict = response.data;

      console.log("Fetched optimizer data:", optimizer_data_dict);

      if (optimizer_data_dict === null) {
        console.error("Invalid optimizer data format:", optimizer_data_dict);
        return rejectWithValue("Invalid optimizer data format");
      }

      const ret: Record<
        string,
        { name: string; value: number | boolean | string | number[] | string[] }
      > = {};
      for (const [key, value] of Object.entries(optimizer_data_dict)) {
        if (typeof value === "number") {
          ret[key] = { name: key, value };
        } else if (typeof value === "string") {
          ret[key] = { name: key, value: value };
        } else if (typeof value === "boolean") {
          ret[key] = { name: key, value: value };
        } else if (
          Array.isArray(value) &&
          value.every((item) => typeof item === "number")
        ) {
          ret[key] = { name: key, value: value as number[] };
        } else if (
          Array.isArray(value) &&
          value.every((item) => typeof item === "string")
        ) {
          ret[key] = { name: key, value: value as string[] };
        } else {
          console.error(`Invalid value for ${key}:`, value);
        }
      }
      return ret;
    } catch (error) {
      console.error("Error fetching training state:", error);
      return rejectWithValue("Failed to fetch training state");
    }
  }
);
