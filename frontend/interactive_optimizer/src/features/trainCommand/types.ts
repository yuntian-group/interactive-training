export interface TrainCommandData {
  command: string;
  args: string;
  time: number;
  uuid: string;
  status: "requested" | "pending" | "running" | "success" | "failed";
}

export default interface trainCommandState {
  commandRecord: Record<string, TrainCommandData>;
  loading: boolean;
  error: string | null;
  response: any | null;
}
