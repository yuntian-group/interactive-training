import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import { getModelInfoFromServer } from "./action";
import type { ModelDataNode, ModelInfoState } from "./type";

const initTree: ModelDataNode = {
  name: "",
  operators: [],
  children: [],
};

const initialState: ModelInfoState = {
  module_tree: initTree,
  status: "idle", // Initial status of the model info fetch
};

const modelInfoStateSlice = createSlice({
  name: "modelInfoState",
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      .addCase(getModelInfoFromServer.pending, (state) => {
        state.status = "loading"; // Set status to loading when the request starts
        state.module_tree = initTree; // Reset the module tree to initial state
        return state;
      })
      .addCase(
        getModelInfoFromServer.fulfilled,
        (state, action: PayloadAction<ModelDataNode>) => {
          const parsedTree = action.payload;
          state.module_tree = parsedTree; // Update state with the parsed tree
          state.status = "succeeded"; // Set status to succeeded if the request is successful
          return state;
        }
      )
      .addCase(getModelInfoFromServer.rejected, (state, action) => {
        console.error("Failed to fetch model info:", action.error.message);
        state.status = "failed"; // Set status to failed if the request fails
        state.module_tree = initTree; // Reset the module tree to initial state
        return state;
      });
  },
});

export default modelInfoStateSlice.reducer;
