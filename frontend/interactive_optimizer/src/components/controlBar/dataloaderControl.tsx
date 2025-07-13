import clsx from "clsx";
import React, { useState } from "react";
import { useAppSelector } from "../../hooks/userTypedHooks";
import { NumericalInputControl, ListInputControl} from "../sharedControl/sharedControl";

const DatasetInfoDisplay: React.FC<{ className?: string }> = ({
  className,
}) => {

  const isTraining = useAppSelector(
    (state) => state.trainInfo.trainInfo.status === "running"
  );

  const datasetInfo = useAppSelector(
    (state) => state.datasetInfoState.datasetInfo
  );

  const initializationParameters = datasetInfo
    ? datasetInfo.initialization_parameters
    : null;
  const interactiveParameters = datasetInfo
    ? datasetInfo.interactive_parameters
    : null;

  
    const [ localInteractiveParameters, setLocalInteractiveParameters ] = useState(interactiveParameters || {});
    const [ localInitializationParameters, setLocalInitializationParameters ] = useState(initializationParameters || {});

  const notRunningInfo = () => {
    return (
      <p className="text-gray-500 p-4 text-sm ">Training is not running.</p>
    );
  };

  const notAvailableInfo = () => {
    return (
      <p className="text-gray-500 p-4 text-sm">
        Dataset information is not available.
      </p>
    );
  };

  const renderParameters = (params: Record<string, any>) => {
    if (!params || Object.keys(params).length === 0) {
      return (
        <p className="text-gray-500 p-4 text-sm">No parameters available.</p>
      );
    } else {
      return Object.entries(params).map(([key, value]) => {
              if (Array.isArray(value)) {
        return (
          <ListInputControl
            key={key}
            id={key + "-input"}
            label={key}
            value={value}
            onChange={(newValue) => {
              console.log(`Updated ${key} to ${newValue}`);
            }}
            className="text-gray-600 bg-white p-2"
          />
        );
      } else if (typeof value === 'number') {
        return (
          <NumericalInputControl
            key={key}
            id={key + "-input"}
            label={key}
            value={value}
            onChange={(newValue) => {
              console.log(`Updated ${key} to ${newValue}`);
            }}
            className="text-gray-600 bg-white p-2"
          />
        );
      } else {
        return (
          <div key={key} className="flex items-center space-x-2 bg-white p-2">
            <span className="font-medium text-gray-700">{key}:</span>
            <span className="text-gray-600">{JSON.stringify(value)}</span>
          </div>
        );
      }
      });
    }
  };

  const displayInitializationParameters = () => {
    if (
      initializationParameters === null ||
      Object.keys(initializationParameters).length === 0
    ) {
      return (
        <p className="text-gray-500 p-4 text-sm ">
          No initialization parameters available.
        </p>
      );
    }
    return renderParameters(initializationParameters);
  };

  const displayInteractiveParameters = () => {
    if (
      interactiveParameters === null ||
      Object.keys(interactiveParameters).length === 0
    ) {
      return (
        <p className="text-gray-500 p-4 text-sm">
          No interactive parameters available.
        </p>
      );
    }
    return renderParameters(interactiveParameters);
  };


  const datasetInfoDisplayAndControls = () => {
    if (!datasetInfo) {
      return notAvailableInfo();
    }

    return (
      <div className="space-y-4 p-4">
        <div>
          <div className="text-sm font-medium text-gray-700 mb-2">
            Interactive Parameters
          </div>
          <div className="interactive-params bg-gray-100 p-4 space-y-2">
            {displayInteractiveParameters()}
            <div className="flex justify-end">
              <button
                className="px-3 py-1 bg-gray-600 text-white text-xs hover:bg-gray-700 transition-colors"
              >
                Apply
              </button>
            </div>
          </div>
        </div>
        <div>
          <div className="text-sm font-medium text-gray-700 mb-2">
            Initialization Parameters
          </div>
          <div className="initialization-params bg-gray-100 p-4 space-y-2">
            {displayInitializationParameters()}
            <div className="flex justify-end">
              <button
                className="px-3 py-1 bg-gray-600 text-white text-xs hover:bg-gray-700 transition-colors"
              >
                Apply
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div
      className={clsx("dataset-info-display h-full flex flex-col", className)}
    >
      <h4 className="flex-shrink-0 flex items-center justify-start text-left text-lg font-semibold p-2 bg-gray-200">
        Dataset Control Panel
      </h4>
      {!isTraining ? notRunningInfo() : datasetInfoDisplayAndControls()}
    </div>
  );
};

export default DatasetInfoDisplay;
