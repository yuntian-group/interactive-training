export interface CheckpointData {
  name: string;
  time: number;
  path: string;
}

export default interface optimizerState {
  state: CheckpointData[];
  loading: boolean;
  error: string | null;
  response: any | null;
}
