export default interface trainCommandState {
  command: string;
  loading: boolean;
  error: string | null;
  response: any | null;
}
