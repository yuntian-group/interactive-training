export default interface TrainLogData {
  steps: number[];
  local_step: number;
  train_log_values: Record<string, number[]>;
}
