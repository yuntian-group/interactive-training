import clsx from "clsx";
import { v4 as uuidv4 } from "uuid";
import React, { useState } from "react";
import { useAppSelector, useAppDispatch } from "../../hooks/userTypedHooks";
import {
  NumericalInputControl,
  ListInputControl,
} from "../sharedControl/sharedControl";

import { postTrainCommand } from "../../features/trainCommand/actions";
import type { TrainCommandData } from "../../features/trainCommand/types";

const DatasetInfoDisplay: React.FC<{ className?: string }> = ({
  className,
}) => {
  const dispatch = useAppDispatch();
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

  const [localInteractiveParameters, setLocalInteractiveParameters] = useState(
    interactiveParameters || {}
  );
  const [localInitializationParameters, setLocalInitializationParameters] =
    useState(initializationParameters || {});

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

  const renderParameters = (
    params: Record<string, any>,
    updateLocalParameters: (params: Record<string, any>) => void
  ) => {
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
                const updatedParams = {
                  [key]: newValue,
                };
                updateLocalParameters(updatedParams);
                console.log(
                  "Updated local interactive parameters:",
                  updatedParams
                );
              }}
              className="text-gray-600 bg-white p-2"
            />
          );
        } else if (typeof value === "number") {
          return (
            <NumericalInputControl
              key={key}
              id={key + "-input"}
              label={key}
              value={value}
              onChange={(newValue) => {
                console.log(`Updated 1 ${key} to ${newValue}`);
                const updatedParams = {
                  [key]: newValue,
                };
                updateLocalParameters(updatedParams);
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

  const displayInitializationParameters = (
    updateLocalInitializationParameters: (params: Record<string, any>) => void
  ) => {
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
    return renderParameters(
      initializationParameters,
      updateLocalInitializationParameters
    );
  };

  const displayInteractiveParameters = (
    updateLocalInteractiveParameters: (params: Record<string, any>) => void
  ) => {
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
    return renderParameters(
      interactiveParameters,
      updateLocalInteractiveParameters
    );
  };

  const updateLocalInteractiveParameters = (params: Record<string, any>) => {
    setLocalInteractiveParameters(() => ({
      ...localInteractiveParameters,
      ...params,
    }));
  };

  const updateLocalInitializationParameters = (params: Record<string, any>) => {
    setLocalInitializationParameters(() => ({
      ...localInitializationParameters,
      ...params,
    }));
  };

  const computeDiffAndDispatchCommand = (
    command: string,
    newState: Record<string, any>,
    oldState: Record<string, any>
  ) => {
    const paramsUpdated: Record<string, any> = {};
    for (const key in newState) {
      if (newState[key] !== oldState[key]) {
        paramsUpdated[key] = newState[key];
      }
    }

    if (Object.keys(paramsUpdated).length === 0) {
      console.log("No parameters updated, skipping command dispatch.");
      return;
    }

    console.log(`Dispatching command: ${command} with params:`, paramsUpdated);

    const trainCommand = {
      command: command,
      args: JSON.stringify(paramsUpdated),
      status: "requested",
      uuid: uuidv4(),
      time: Date.now(),
    } as TrainCommandData;
    dispatch(postTrainCommand(trainCommand));
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
            {displayInteractiveParameters(updateLocalInteractiveParameters)}
            <div className="flex justify-end">
              <button
                onClick={() =>
                  computeDiffAndDispatchCommand(
                    "update_dataset_runtime_hyperparameters",
                    localInteractiveParameters,
                    interactiveParameters || {}
                  )
                }
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
            {displayInitializationParameters(
              updateLocalInitializationParameters
            )}
            <div className="flex justify-end">
              <button
                onClick={() =>
                  computeDiffAndDispatchCommand(
                    "update_dataset",
                    localInitializationParameters,
                    initializationParameters || {}
                  )
                }
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
