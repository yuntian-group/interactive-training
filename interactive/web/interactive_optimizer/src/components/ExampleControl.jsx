import React, { useEffect, useRef, useState } from "react";

function SubscribePanel( {ws} ) {
    const [runId, setRunId] = useState("");
    const handleSubscribe = (runId) => {
        if (!runId) return;
        ws.send(JSON.stringify({ "action": "subscribe", "run_id": runId }));
    };

    return (
        <div className="flex items-center gap-2">
        <input placeholder="Enter Run Id" className="w-48" onChange={
            (e) => setRunId(e.target.value) } value={runId}/>
        <button variant="outline" size="sm" onClick={() => {
            handleSubscribe(runId);
        }}>Subscribe</button>
        </div>
    );
}

function CommandPanel({ host, runId }){
    const [command, setCommand] = useState("");

    const handleSendCommand = async () => {
        console.log("send command")
        if (!command || !runId) return;
        try {
            await fetch(`http://localhost:9876/api/v1/runs/${runId}/command`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ command }),
            });
            setCommand(""); // Clear input after sending
        } catch (err) {
            console.error("Command failed", err);
        }
    };

    return (
        <div className="flex items-center gap-2">
            <input value={command} onChange={(e) => setCommand(e.target.value)} placeholder="Enter command" className="flex-grow" />
            <button onClick={handleSendCommand} className="p-2 bg-blue-500 text-white rounded">
                SEND
            </button>
        </div>
    );
}

/**
 * Props
 * -----
 * runId  : string  – UUID returned by POST /api/v1/runs/new
 * host   : string  – host:port where the FastAPI server lives (no protocol).
 */
export default function RunDashboard({ host }) {
  const [messages, setMessages] = useState([]);       // all WS updates
  const [command, setCommand] = useState("");        // text in input box
  const [status, setStatus]   = useState("disconnected");
  const [runId, setRunId] = useState(null); // current run ID
  const wsRef = useRef(null);

  

  /** Establish / tear‑down the client‑websocket. */
  useEffect(() => {
    console.log(host);
    wsRef.current =  new WebSocket(host);

    wsRef.current.onopen = () => {
      setStatus("connected");
    };

    wsRef.current.onmessage = (evt) => {
      try {
        const parsed = JSON.parse(evt.data);
        setMessages((prev) => [...prev, parsed]);
        handleCurrentMessage(parsed);
      } catch (_) {
        setMessages((prev) => [...prev, { type: "raw", data: evt.data }]);
      }
    };

    wsRef.current.onclose = () => setStatus("disconnected");
    wsRef.current.onerror  = () => setStatus("error");

    return () => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
          wsRef.current.close();
        }
    };
  }, [host]);

  const handleCurrentMessage = (msg) => {
        console.log("Received message:", msg);
        if (msg.action === "subscribe" && msg.run_id) {
            setRunId(msg.run_id);
            console.log("Subscribed to run:", msg.run_id);
            console.log(status);
            console.log("WebSocket status:", wsRef.current.readyState);
            console.log(runId);
        } 
    }

  /** POST a command to the worker via HTTP. */
  const sendCommand = async (cmd) => {
    if (!cmd || !runId) return;
    try {
      await fetch(`http://${host}/api/v1/runs/${runId}/command`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: cmd }),
      });
    } catch (err) {
      console.error("command failed", err);
    }
    setCommand("");
  };

  /** Convenience handlers for preset buttons. */
  const preset = (c) => () => sendCommand(c);

  return (
    <div className="flex flex-col gap-4 p-4 max-w-3xl mx-auto">
      {/* Status header */}
      <div>
        <div className="flex items-center justify-between p-4">
          <span className="font-semibold text-lg">Run ID:</span>
          <code className="text-xs truncate max-w-md">{runId || "–"}</code>
          <span className={`px-2 py-1 rounded text-xs ${status === "connected" ? "bg-green-100 text-green-800" : status === "error" ? "bg-red-100 text-red-800" : "bg-gray-100 text-gray-600"}`}>{status}</span>
        </div>
      </div>


      {/* if run id not available render subscribe panel, otherwise render command panel */}
        {status === "disconnected" || status === "error" ? (
            <div className="text-red-500">WebSocket disconnected. Please check the server.</div>
        ) : (!runId ? (
            <SubscribePanel ws={wsRef.current} />
        ) : (
            <CommandPanel host={host} runId={runId} />
        ))}

      {/* Message log */}
      <div>
        <div className="overflow-y-auto max-h-96 p-4 space-y-2 text-sm font-mono bg-black/90 text-gray-100 rounded-lg">
          {messages.length === 0 && <div className="italic text-gray-400">No messages yet…</div>}
          {messages.map((m, idx) => (
            <div key={idx} className="break-all">
              {typeof m === "string" ? m : JSON.stringify(m)}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
