export interface TrainInfoData {
  start_time: number;
  status: string;
}

export default interface trainInfoState {
  trainInfo: TrainInfoData;
  loading: boolean;
  error: string | null;
}
