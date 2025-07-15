import clsx from "clsx";
import { v4 as uuidv4 } from "uuid";
import { useState } from "react";
import type { CheckpointData } from "../../features/checkpointState/type";
import { postTrainCommand } from "../../features/trainCommand/actions";
import type { TrainCommandData } from "../../features/trainCommand/types";
import { useAppSelector, useAppDispatch } from "../../hooks/userTypedHooks";

const SingleCheckpointDisplay: React.FC<{
  checkpoint: CheckpointData;
  handleLoadCheckpoint?: (checkpoint: CheckpointData, branchName: string) => void;
}> = ({ checkpoint, handleLoadCheckpoint }) => {
  const [showPopup, setShowPopup] = useState(false);
  const [branchName, setBranchName] = useState("newBranch");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (handleLoadCheckpoint && branchName.trim()) {
      handleLoadCheckpoint(checkpoint, branchName);
      setShowPopup(false);
      setBranchName("newBranch");
    }
  };

  return (
    <>
      <div className="checkpoint-display p-2 border border-gray-200 rounded-none bg-gray-50 flex-shrink-0">
        <h3 className="font-semibold text-sm">{checkpoint.checkpoint_dir}</h3>
        <p className="text-xs text-gray-600">
          Created at: {new Date(checkpoint.time * 1000).toLocaleString()}
        </p>
        <p className="text-xs text-gray-600">
          Description: {checkpoint.checkpoint_dir}
        </p>
        <button
          onClick={() => setShowPopup(true)}
          className={clsx(
            "mt-1 px-2 py-1 text-xs focus:outline-none transition-colors duration-200 bg-gray-600 hover:bg-gray-700 text-white"
          )}
        >
          Load Checkpoint
        </button>
      </div>

      {/* Popup Modal */}
      {showPopup && (
        <div className="fixed inset-0 bg-blue-100/50 flex items-center justify-center z-50">
          <div className="bg-white p-6 rounded-none w-80">
            <h3 className="text-lg font-semibold mb-4">Load Checkpoint</h3>
            <form onSubmit={handleSubmit}>
              <label className="block text-sm font-medium mb-2">
                Branch Name:
              </label>
              <input
                type="text"
                value={branchName}
                onChange={(e) => setBranchName(e.target.value)}
                className="w-full p-2 border border-gray-300 focus:outline-none focus:ring-2 focus:round-none focus:ring-back-500"
                placeholder="Enter branch name"
                autoFocus
              />
              <div className="flex justify-end gap-2 mt-4">
                <button
                  type="button"
                  onClick={() => {
                    setShowPopup(false);
                    setBranchName("");
                  }}
                  className="px-4 py-2 text-gray-600 border border-gray-300 hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!branchName.trim()}
                  className="px-4 py-2 bg-blue-500 text-white hover:bg-blue-600 disabled:bg-gray-400"
                >
                  Load
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};

const CheckpointControl: React.FC<{
  className?: string;
}> = ({ className }) => {
  const checkpointState = useAppSelector((state) => state.checkpointState);
  const isTraining = useAppSelector(
    (state) => state.trainInfo.trainInfo.status === "running"
  );
  const dispatch = useAppDispatch();

  const handleSaveCurrentState = () => {
    dispatch(
      postTrainCommand({
        command: "save_checkpoint",
        uuid: uuidv4(),
        args: "",
        time: Date.now(),
        status: "requested",
      } as TrainCommandData)
    );
  };

  const handleLoadCheckpoint = (checkpoint: CheckpointData, branchName: string) => {
    dispatch(
      postTrainCommand({
        command: "load_checkpoint",
        uuid: uuidv4(),
        args: JSON.stringify({ uuid: checkpoint.uuid, branch_name: branchName}),
        time: Date.now(),
        status: "requested",
      } as TrainCommandData)
    );
  };

  return (
    <div
      className={clsx("checkpoint-control h-full flex flex-col", className)}
    >
      <h4 className="flex-shrink-0 flex items-center justify-start text-left text-lg font-semibold p-2 bg-gray-200">
        Checkpoint Management
      </h4>
      {/* Save Checkpoint Panel */}
      <div className="save-checkpoint p-3 mb-2  bg-gradient-to-r from-gray-50 to-gray-100 shadow-sm h-20 flex-shrink-0 justify-center items-center flex flex-col">
        <button
          onClick={() => {
            handleSaveCurrentState();
          }}
          disabled={!isTraining}
          className="rounded-lg text-lg bg-green-500 hover:bg-green-600 disabled:bg-gray-400 disabled:hover:bg-gray-400 text-white focus:outline-none focus:ring-2 focus:ring-green-300 disabled:focus:ring-0 transition-all duration-200 shadow-sm disabled:shadow-none flex items-center justify-center h-4/5 w-4/5 font-medium disabled:cursor-not-allowed"
        >
          Save Current State
        </button>
      </div>

      {/* Current Checkpoint List */}
      <div className="checkpoint-list flex flex-col flex-1 min-h-0 p-4">
        <h2 className="text-base font-semibold mb-1 h-6 flex-shrink-0">
          Current Checkpoint
        </h2>
        <div className="flex-1 overflow-y-auto">
          {checkpointState.state.length > 0 ? (
            <div className="space-y-1">
              {checkpointState.state.map((checkpoint) => (
                <SingleCheckpointDisplay
                  key={checkpoint.uuid}
                  checkpoint={checkpoint}
                  handleLoadCheckpoint={(ckpt, branchName) => handleLoadCheckpoint(ckpt, branchName)}
                />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-sm">No checkpoints available.</p>
          )}
        </div>
      </div>
    </div>
  );
};
export default CheckpointControl;
