export interface TrainCommandData {
  uuid: string;
  command: string;
  args: string;
  time: number;
  status: "requeted" | "pending" | "running" | "success" | "failed";
}

export default interface trainCommandState {
  commandRecord: Record<string, TrainCommandData>;
  loading: boolean;
  error: string | null;
  response: any | null;
}
