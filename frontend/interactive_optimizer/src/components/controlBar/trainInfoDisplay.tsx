import clsx from "clsx";
import { v4 as uuidv4 } from "uuid";
import type { TrainCommandData } from "../../features/trainCommand/types";
import { postTrainCommand } from "../../features/trainCommand/actions";
import { useAppSelector, useAppDispatch } from "../../hooks/userTypedHooks";
import { Pause, Square, BarChart3, Play } from "lucide-react";

const TrainInfoDisplay: React.FC<{
  className?: string;
}> = ({ className }) => {
  const trainInfo = useAppSelector((state) => state.trainInfo.trainInfo);
  const loading = useAppSelector((state) => state.trainInfo.loading);
  const error = useAppSelector((state) => state.trainInfo.error);

  const makeCommandForTrain = (command: string, args: string = "") => {
    return {
      command: command,
      args: args,
      status: "requested",
      uuid: uuidv4(),
      time: Date.now(),
    } as TrainCommandData;
  };

  const dispatch = useAppDispatch();

  const handlePauseTraining = () => {
    dispatch(postTrainCommand(makeCommandForTrain("pause_training")));
  };

  const handleResumeTraining = () => {
    dispatch(postTrainCommand(makeCommandForTrain("resume_training")));
  };

  const handleStopTraining = () => {
    dispatch(postTrainCommand(makeCommandForTrain("stop_training")));
  };

  const handleEvaluate = () => {
    dispatch(postTrainCommand(makeCommandForTrain("do_evaluate")));
  };

  return (
    <div className={clsx("train-info-display", className)}>
      <h4 className="flex-shrink-0 flex items-center justify-start text-left text-lg font-semibold p-2 bg-gray-200">
        Training Information
      </h4>
      {loading ? (
        <p className="p-4 text-sm text-gray-500">Loading...</p>
      ) : error ? (
        <p className="p-4 text-sm text-gray-500">
          {"Fail to load train information: " + error}
        </p>
      ) : (
        <div className="space-y-4 p-4">
          <div className="space-y-2">
            <p>
              <strong>Run Name:</strong> {"TEST NAME"}
            </p>
            <p>
              <strong>Start Time:</strong>{" "}
              {new Date(trainInfo.start_time * 1000).toLocaleString()}
            </p>
            <p>
              <strong>Status:</strong> {trainInfo.status}
            </p>
          </div>

          <div className="grid grid-cols-2 gap-2">
            {trainInfo.status === "paused" ? (
              <button
                onClick={handleResumeTraining}
                className="px-3 py-2 text-sm bg-green-500 text-white rounded hover:bg-green-600 flex items-center justify-center space-x-1"
              >
                <Play size={16} />
                <span>Resume</span>
              </button>
            ) : (
              <button
                onClick={handlePauseTraining}
                className="px-3 py-2 text-sm bg-yellow-500 text-white rounded hover:bg-yellow-600 disabled:opacity-50 flex items-center justify-center space-x-1"
                disabled={trainInfo.status !== "running"}
              >
                <Pause size={16} />
                <span>Pause</span>
              </button>
            )}
            <button
              onClick={handleStopTraining}
              className="px-3 py-2 text-sm bg-red-500 text-white rounded hover:bg-red-600 disabled:opacity-50 flex items-center justify-center space-x-1"
              disabled={trainInfo.status === "stopped"}
            >
              <Square size={16} />
              <span>Stop</span>
            </button>
            <button
              onClick={handleEvaluate}
              className="px-3 py-2 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 flex items-center justify-center space-x-1 col-span-2"
              disabled={trainInfo.status === "stopped"}
            >
              <BarChart3 size={16} />
              <span>Evaluate</span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};
export default TrainInfoDisplay;
