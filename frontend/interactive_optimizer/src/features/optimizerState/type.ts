export interface OptimizerData {
  name: string;
  value: number | boolean | string | number[] | string[];
}

export default interface optimizerState {
  optimizer_state: Record<string, OptimizerData>;
  loading: boolean;
  error: string | null;
  response: any | null;
}
