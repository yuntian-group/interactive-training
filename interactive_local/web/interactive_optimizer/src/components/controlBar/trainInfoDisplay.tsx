import clsx from "clsx";
import { useAppSelector } from "../../hooks/userTypedHooks";

const TrainInfoDisplay: React.FC<{
  className?: string;
}> = ({ className }) => {
  const trainInfo = useAppSelector((state) => state.trainInfo.trainInfo);
  const loading = useAppSelector((state) => state.trainInfo.loading);
  const error = useAppSelector((state) => state.trainInfo.error);

  return (
    <div className={clsx("train-info-display p-4", className)}>
      <h2 className="text-lg font-semibold mb-2">Training Information</h2>
      {loading ? (
        <p>Loading...</p>
      ) : error ? (
        <p className="text-red-500">{error}</p>
      ) : (
        <div className="space-y-2">
          <p>
            <strong>Start Time:</strong>{" "}
            {new Date(trainInfo.start_time).toLocaleString()}
          </p>
          <p>
            <strong>Status:</strong> {trainInfo.status}
          </p>
        </div>
      )}
    </div>
  );
};
export default TrainInfoDisplay;
