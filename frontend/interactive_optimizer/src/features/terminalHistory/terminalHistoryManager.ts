class TermnialHistoryManager {
  public history: string[] = [];
  private static instance: TermnialHistoryManager | null = null;
  private static MAX_HISTORY_SIZE = 5000; // Optional: Limit the history size

  public static getInstance(): TermnialHistoryManager {
    if (!TermnialHistoryManager.instance) {
      TermnialHistoryManager.instance = new TermnialHistoryManager();
    }
    return TermnialHistoryManager.instance;
  }

  public addToHistory(message: string): void {
    this.history.push(message);
    // Optionally, you can limit the history size
    if (this.history.length > TermnialHistoryManager.MAX_HISTORY_SIZE) {
      this.history.shift(); // Remove the oldest message
    }
  }
}
export default TermnialHistoryManager;
