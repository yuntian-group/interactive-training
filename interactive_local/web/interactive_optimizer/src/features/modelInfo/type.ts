export interface ModelDataNode {
  name: string;
  operators: string[];
  children: ModelDataNode[];
}

export interface ModelInfoState {
  module_tree: ModelDataNode;
  status: "idle" | "loading" | "succeeded" | "failed"; // Status of the model info fetch
}
