export interface OptimizerData {
  name: string;
  value: number;
}

export default interface optimizerState {
  optimizer_state: Record<string, OptimizerData>;
  loading: boolean;
  error: string | null;
  response: any | null;
}
