import clsx from "clsx";
import { useAppSelector } from "../../hooks/userTypedHooks";

const TrainInfoDisplay: React.FC<{
  className?: string;
}> = ({ className }) => {
  const trainInfo = useAppSelector((state) => state.trainInfo.trainInfo);
  const loading = useAppSelector((state) => state.trainInfo.loading);
  const error = useAppSelector((state) => state.trainInfo.error);

  console.log(trainInfo.start_time);

  return (
    <div className={clsx("train-info-display", className)}>
      <h4 className="flex-shrink-0 flex items-center justify-start text-left text-lg font-semibold p-2 bg-gray-200">
        Training Information
      </h4>
      {loading ? (
        <p className="p-4 text-sm text-gray-500">Loading...</p>
      ) : error ? (
        <p className="p-4 text-sm text-gray-500">{"Fail to load train information: "+ error}</p>
      ) : (
        <div className="space-y-2">
          <p>
            <strong>Start Time:</strong>{" "}
            {new Date(trainInfo.start_time * 1000).toLocaleString()}
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
