import clsx from "clsx";
import React, { useState, useEffect } from "react";
import type { CheckpointData } from "../../features/checkpointState/type";

const SingleCheckpointDisplay: React.FC<{
    checkpoint: CheckpointData;
    onLoadCheckpoint?: (checkpoint: CheckpointData) => void;
}> = ({ checkpoint, onLoadCheckpoint }) => {
    const handleLoadCheckpoint = () => {
        if (onLoadCheckpoint) {
            onLoadCheckpoint(checkpoint);
        }
    };

    return (
        <div className="checkpoint-display p-2 border rounded-none bg-gray-50">
          <h3 className="font-semibold">{checkpoint.name}</h3>
          <p className="text-sm text-gray-600">
            Created at: {new Date(checkpoint.time).toLocaleString()}
          </p>
          <p className="text-sm text-gray-600">Description: {checkpoint.path}</p>
          <button
            onClick={onLoadCheckpoint ? handleLoadCheckpoint : undefined}
            disabled={!onLoadCheckpoint}
            className={clsx(
              "mt-2 px-3 py-1 rounded text-sm focus:outline-none transition-colors duration-200 bg-blue-500 hover:bg-blue-600 text-white"
            )}
          >
            Load Checkpoint
          </button>
        </div>
    );
};

const CheckpointControl: React.FC<{
  className?: string;
}> = ({ className }) => {
  return (
    <div className={clsx("checkpoint-control p-4", className)}>
      <h2 className="text-lg font-semibold mb-2">Current Checkpoint</h2>

      <div className="checkpoint-list space-y-2">
        {/* Example checkpoints, replace with dynamic data */}
        <SingleCheckpointDisplay
          checkpoint={{
            name: "Checkpoint 1",
            time: Date.now(),
            path: "/path/to/checkpoint1",
          }}
        />
      </div>
    </div>
  );
};
export default CheckpointControl;
