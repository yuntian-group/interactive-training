import { createSlice, type PayloadAction } from "@reduxjs/toolkit";
import { getModelInfoFromServer } from "./action";
import type { ModelDataNode, ModelInfoState } from "./type";

const initTree: ModelDataNode = {
  name: "",
  operators: [],
  hyperparameters: [],
  children: [],
  moduleType: "Unknown", // Default module type
};

const initialState: ModelInfoState = {
  moduleTree: initTree,
  nodeMap: {}, // Initially empty map for module nodes
  status: "idle", // Initial status of the model info fetche
  selectedLayer: "", // Initially no layer is selected
};

const modelInfoStateSlice = createSlice({
  name: "modelInfoState",
  initialState,
  reducers: {
    selectLayer: (state, action: PayloadAction<string>) => {
      console.log("Reducer Selecting layer:", action.payload);
      state.selectedLayer = action.payload; // Update the selected layer in the state
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(getModelInfoFromServer.pending, (state) => {
        state.status = "loading"; // Set status to loading when the request starts
        state.moduleTree = initTree; // Reset the module tree to initial state
        return state;
      })
      .addCase(
        getModelInfoFromServer.fulfilled,
        (
          state,
          action: PayloadAction<[ModelDataNode, Record<string, ModelDataNode>]>
        ) => {
          const [parsedTree, nodeMap] = action.payload;
          state.moduleTree = parsedTree; // Update state with the parsed tree
          state.status = "succeeded"; // Set status to succeeded if the request is successful
          state.nodeMap = nodeMap; // Update the nodeMap with the fetched data
          return state;
        }
      )
      .addCase(getModelInfoFromServer.rejected, (state, action) => {
        console.error("Failed to fetch model info:", action.error.message);
        state.status = "failed"; // Set status to failed if the request fails
        state.moduleTree = initTree; // Reset the module tree to initial state
        return state;
      });
  },
});

export const { selectLayer } = modelInfoStateSlice.actions;

export default modelInfoStateSlice.reducer;
