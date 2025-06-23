import { createAsyncThunk } from "@reduxjs/toolkit";
import { getModelInfo } from "../../api/api";
import type { ModelDataNode } from "./type";

const parseModelTreeInit = (
  module_tree: Record<string, any>
): ModelDataNode => {
  const parseNode = (node: Record<string, any>): ModelDataNode => {
    return {
      name: node.name || "",
      operators: node.operators || [],
      children: (node.children || []).map(parseNode),
    };
  };
  return parseNode(module_tree);
};

export const getModelInfoFromServer = createAsyncThunk(
  "model/getModelInfoFromServer",
  async (_, { rejectWithValue }) => {
    try {
      const response = await getModelInfo();
      const moduleTree = response.data;
      console.log("Fetched model data:", moduleTree);
      // Transform the module_tree to match ModelDataNode structure
      try {
        if (!moduleTree || typeof moduleTree !== "object") {
          throw new Error("Invalid module tree structure");
        }
        const parsedTree = parseModelTreeInit(moduleTree);
        return parsedTree;
      } catch (error) {
        console.error("Error validating module tree structure:", error);
        return rejectWithValue("Invalid module tree structure");
      }
    } catch (error) {
      console.error("Error fetching model info:", error);
      return rejectWithValue("Failed to fetch model info");
    }
  }
);
