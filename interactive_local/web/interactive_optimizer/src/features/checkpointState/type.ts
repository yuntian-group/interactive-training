export interface CheckpointData {
  time: number;
  checkpoint_dir: string;
  uuid: string;
  branch_id: string;
  global_step: number;
}

export default interface optimizerState {
  state: CheckpointData[];
  loading: boolean;
  error: string | null;
  response: any | null;
}
