import { useEffect, useRef } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { useAppSelector } from "../../hooks/userTypedHooks";
import "xterm/css/xterm.css";
import TerminalHistoryManager from "../../features/terminalHistory/terminalHistoryManager";

const LogDisplay: React.FC<{ className?: string }> = ({}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const terminalRef = useRef<Terminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const historyLoadedRef = useRef<boolean>(false);

  const curLog = useAppSelector((state) => state.trainLogData.curLog);
  const curLogVersion = useAppSelector(
    (state) => state.trainLogData.localLogVersion
  );

  const terminalHistoryManager = TerminalHistoryManager.getInstance();

  useEffect(() => {
    if (!containerRef.current) return;

    // Initialize terminal
    const terminal = new Terminal({
      cursorBlink: true,
      fontSize: 12,
      theme: {
        background: "#ffffff",
        foreground: "#1e1e1e",
      },
    });
    const fitAddon = new FitAddon();

    terminal.loadAddon(fitAddon);
    terminal.open(containerRef.current);

    // delay initial fit until internal services are up
    window.requestAnimationFrame(() => {
      try {
        // only fit if proposeDimensions() returns a valid object
        (fitAddon as any).proposeDimensions?.() && fitAddon.fit();
      } catch {
        /* swallow */
        console.warn("Error fitting terminal on initial load");
      }
    });

    terminal.writeln("Welcome to Interactive Trainer");
    terminalRef.current = terminal;
    fitAddonRef.current = fitAddon;
    const historyLength = terminalHistoryManager.history.length
    terminalHistoryManager.history.slice(0, historyLength - 1).forEach((message) => {
      terminal.writeln(message);
    });
    terminal.scrollToBottom();
    historyLoadedRef.current = true;

    // Resize observer
    let resizeTimer: number;
    const observer = new ResizeObserver(() => {
      clearTimeout(resizeTimer);
      const el = containerRef.current;
      if (!el) return;
      resizeTimer = window.setTimeout(() => {
        const fit = fitAddonRef.current;
        // call fit only if private proposeDimensions() yields valid cols/rows
        const dims = (fit as any)?.proposeDimensions?.();
        if (dims?.cols > 0 && dims?.rows > 0) {
          try {
            fit!.fit();
          } catch (error) {
            console.warn("Error fitting terminal:", error);
          }
        }
      }, 100);
    });
    observer.observe(containerRef.current);

    return () => {
      observer.disconnect();
      clearTimeout(resizeTimer);
      terminal.dispose();
      terminalRef.current = null;
      fitAddonRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (terminalRef.current && historyLoadedRef.current && curLog) {
      // append each incoming line rather than clearing
      terminalRef.current.writeln(curLog);
      terminalRef.current.scrollToBottom();
    }
  }, [curLog, curLogVersion]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100%",
        minWidth: 0,
        minHeight: 0,
      }}
    />
  );
};

export default LogDisplay;
