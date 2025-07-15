export interface SingleSignature {
  name: string; // Name of the operation
  type: string; // Type of the operation, e.g., "int", "float", "str", etc.
  value: any; // Value of the operation
  min?: number; // Minimum value for numerical types
  max?: number; // Maximum value for numerical types
  step?: number; // Step size for numerical types
}

export interface ModelOperation {
  name: string; // Name of the operation
  signature: SingleSignature[]; // List of signatures for the operation
}

export interface ModelHyperparameter {
  name: string; // Name of the hyperparameter
  value: any; // Value of the hyperparameter
  type: string; // Type of the hyperparameter, e.g., "int", "float", "str", etc.
}

export interface ModelDataNode {
  name: string;
  operators: ModelOperation[];
  hyperparameters: ModelHyperparameter[];
  children: ModelDataNode[];
  moduleType: string;
}

export interface ModelInfoState {
  moduleTree: ModelDataNode;
  status: "idle" | "loading" | "succeeded" | "failed"; // Status of the model info fetch
  selectedLayer: string; // Currently selected layer in the model info
  nodeMap: Record<string, ModelDataNode>; // Map of module names to ModelDataNode
}
