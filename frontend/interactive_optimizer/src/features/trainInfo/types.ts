export interface TrainInfoData {
  startTime: number;
  status: string;
  runName: string;
}

export default interface trainInfoState {
  trainInfo: TrainInfoData;
  loading: boolean;
  error: string | null;
}
