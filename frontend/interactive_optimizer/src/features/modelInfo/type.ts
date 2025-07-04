export interface ModelDataNode {
  name: string;
  operators: string[];
  children: ModelDataNode[];
  moduleType: string;
}

export interface ModelInfoState {
  moduleTree: ModelDataNode;
  status: "idle" | "loading" | "succeeded" | "failed"; // Status of the model info fetch
  selectedLayer: string; // Currently selected layer in the model info
  nodeMap: Record<string, ModelDataNode>; // Map of module names to ModelDataNode
}
