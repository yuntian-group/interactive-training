import { createAsyncThunk } from "@reduxjs/toolkit";
import { getModelInfo } from "../../api/api";
import type { ModelDataNode } from "./type";

const parseModelTreeInit = (
  moduleTree: Record<string, any>,
  moduleMap: Record<string, ModelDataNode>
): ModelDataNode => {
  const parseNode = (
    node: Record<string, any>,
    moduleMap: Record<string, ModelDataNode>
  ): ModelDataNode => {
    const nodeData = {
      name: node.name || "",
      operators: node.operators || [],
      // explicitly pass moduleMap so it stays in scope
      children: (node.children || []).map((child: Record<string, any>) =>
        parseNode(child, moduleMap)
      ),
      moduleType: node.module_type || "Unknown", // Default to "Unknown" if not specified
    } as ModelDataNode;

    moduleMap[nodeData.name] = nodeData;
    return nodeData;
  };
  return parseNode(moduleTree, moduleMap);
};

export const getModelInfoFromServer = createAsyncThunk<
  [ModelDataNode, Record<string, ModelDataNode>],
  void
>("model/getModelInfoFromServer", async (_, { rejectWithValue }) => {
  try {
    const response = await getModelInfo();
    const moduleTree = response.data;
    const moduleDict: Record<string, ModelDataNode> = {};
    console.log("Fetched model data:", moduleTree);
    // Transform the module_tree to match ModelDataNode structure
    try {
      if (!moduleTree || typeof moduleTree !== "object") {
        throw new Error("Invalid module tree structure");
      }
      const parsedTree = parseModelTreeInit(moduleTree, moduleDict);
      return [parsedTree, moduleDict];
    } catch (error) {
      console.error("Error validating module tree structure:", error);
      return rejectWithValue("Invalid module tree structure");
    }
  } catch (error) {
    console.error("Error fetching model info:", error);
    return rejectWithValue("Failed to fetch model info");
  }
});
