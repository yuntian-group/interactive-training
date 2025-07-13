export interface DatasetInfo {
  initialization_parameters: Record<string, any>;
  interactive_parameters: Record<string, any>;
}

export interface DatasetInfoState {
  datasetInfo: DatasetInfo | null; // Holds the dataset information
  status: "idle" | "loading" | "succeeded" | "failed";
  error?: string; // Optional error message in case of failure
}
