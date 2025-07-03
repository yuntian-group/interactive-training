import clsx from "clsx";
import { useAppSelector } from "../../hooks/userTypedHooks";
import MetricsDisplayChart from "./metricsDisplayChart";
import { useMemo } from "react";
import branchMetricsBuffer from "../../features/trainLogData/logBuffers";

type Props = React.HTMLAttributes<HTMLDivElement>;

const buildSeriesForDisplay = (
  displayBranches: string[]
): [string[], string[][], number[][][][]] => {
  // group metrics by branchId

  console.log("Building series for display with branches:", displayBranches);

  let metricsNames: string[] = [];
  let branchNames: string[][] = [];
  let dataOptions: number[][][][] = [];
  for (const metricsName of branchMetricsBuffer.keys()) {
    let curBranchNames: string[] = [];
    let curDataOptions: number[][][] = [];
    const branchMetrics = branchMetricsBuffer.get(metricsName);
    if (!branchMetrics) {
      continue; // skip if no metrics for this name
    }
    for (const branchName of branchMetrics.keys()) {
      let curBranchDataOptions: number[][] = [];
      if (!displayBranches.includes(branchName)) {
        continue; // skip branches not in displayBranches
      }
      const branchData = branchMetrics.get(branchName);
      if (!branchData) continue;
      curBranchNames.push(branchName);
      const globalSteps = branchData.globalSteps.subarray(0, branchData.len);
      const vals = branchData.vals.subarray(0, branchData.len);
      globalSteps.forEach((step, index) => {
        const tp = [step, vals[index]];
        curBranchDataOptions.push(tp);
      });
      curDataOptions.push(curBranchDataOptions);
    }
    if (curBranchNames.length > 0 && curDataOptions.length > 0) {
      metricsNames.push(metricsName);
      branchNames.push(curBranchNames);
      dataOptions.push(curDataOptions);
    }
  }
  return [metricsNames, branchNames, dataOptions];
};

const MetricsPanel: React.FC<Props> = ({ className }: Props) => {
  const localDataVersion = useAppSelector(
    (state) => state.trainLogData.localDataVersion
  );

  const displayBranch = useAppSelector(
    (state) => state.trainLogData.displayBranch
  );

  const [metricsNames, branchNames, dataOptions] = useMemo(() => {
    // This will trigger a re-render when localDataVersion changes
    return buildSeriesForDisplay(displayBranch);
  }, [localDataVersion, displayBranch]);


  return (
    <div
      className={clsx(
        "bg-gray-100 text-black p-4 flex flex-wrap overflow-auto font-sans font-semibold",
        className
      )}
    >
      {metricsNames.length > 0 ? (
        metricsNames.map((metricName, index) => (
          <div
            key={metricName}
            className="h-[33vh] max-h-[33vh] w-full md:w-1/2 lg:w-1/2 xl:w-1/3 flex-shrink-0 p-1"
          >
            <MetricsDisplayChart
              className="border-gray-200 p-4 h-full w-full flex-shrink-0 border bg-white hover:bg-gray-50 transition-colors duration-200"
              title={metricName}
              branchNames={branchNames[index]}
              branchDataList={dataOptions[index]}
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
