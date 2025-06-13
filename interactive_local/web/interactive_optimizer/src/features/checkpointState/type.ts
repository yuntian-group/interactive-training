export interface CheckpointData {
  time: number;
  checkpoint_dir: string;
  uuid: string;
}

export default interface optimizerState {
  state: CheckpointData[];
  loading: boolean;
  error: string | null;
  response: any | null;
}
