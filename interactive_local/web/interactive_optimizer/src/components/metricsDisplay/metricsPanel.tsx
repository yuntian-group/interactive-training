import clsx from "clsx";
import { useAppSelector } from "../../hooks/userTypedHooks";
import MetricsDisplayChart from "./metricsDisplayChart";

type Props = React.HTMLAttributes<HTMLDivElement>;

const MetricsPanel: React.FC<Props> = ({ className }: Props) => {
    const trainSteps = useAppSelector((state) => state.trainLogData.steps);
    const trainLogValues = useAppSelector(
        (state) => state.trainLogData.train_log_values
    );

    console.log("init steps", trainSteps);
    console.log("init log value", trainLogValues);

    const names = Object.keys(trainLogValues).sort();

    return (
        <div
            className={clsx(
                "bg-gray-100 text-black p-4 flex flex-wrap overflow-auto font-sans font-semibold",
                className
            )}
        >
            {names.length > 0 ? (
                names.map((name) => (
                    <div
                        key={name}
                        className="h-[33vh] max-h-[33vh] w-1/2 lg:w-1/2 xl:w-1/3 flex-shrink-0 p-1"
                    >
                        <MetricsDisplayChart
                            xAxisData={trainSteps}
                            yAxisData={trainLogValues[name]}
                            title={name}
                            className="border-gray-200 p-4 h-full w-full flex-shrink-0 border bg-white hover:bg-gray-50 transition-colors duration-200"
                        />
                    </div>
                ))
            ) : (
                <div className="text-gray-500 text-center">
                    No metrics data available.
                </div>
            )}
        </div>
    );
};

export default MetricsPanel;
