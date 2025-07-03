export interface SingleMetricsPoint {
  local_step: number;
  wall_time: number;
  branch_id: string;
  metrics: Record<string, number>;
}

export interface BranchInfo {
  id: string;
  wall_time: number;
  parent: string;
}

export default interface TrainLogData {
  localDataVersion: number;
  currentBranch: string;
  branchInfo: Record<string, BranchInfo>;
  branchTree: Record<string, string[]>;
  displayBranch: string[];
}
