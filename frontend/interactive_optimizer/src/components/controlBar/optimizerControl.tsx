import clsx from "clsx";
import { v4 as uuidv4 } from "uuid";
import React, { useState, useEffect } from "react";
import { useAppDispatch, useAppSelector } from "../../hooks/userTypedHooks";
import { Box, Typography } from "@mui/material";
import type { OptimizerData } from "../../features/optimizerState/type";
import type { TrainCommandData } from "../../features/trainCommand/types";
import { postTrainCommand } from "../../features/trainCommand/actions";
import {
  NumericalInputControlWithSlider,
  BoolControlWithSwitch,
  FixedLengthNumericalListControl,
  KeyValueDisplay
} from "../sharedControl/sharedControl";

const generateOptimizerUpdateTrainCommand = (
  paramsUpdated: Record<string, OptimizerData>
): TrainCommandData => {
  const command: TrainCommandData = {
    command: "update_optimizer",
    args: JSON.stringify(paramsUpdated),
    uuid: uuidv4(),
    time: Date.now(),
    status: "requested",
  };
  return command;
};

type Props = React.HTMLAttributes<HTMLDivElement>;

const OptimizerControl: React.FC<Props> = ({ className }: Props) => {
  const optimizerStateServer: Record<string, OptimizerData> = useAppSelector(
    (state) => state.optimizerState.optimizer_state
  );

  const trainStatus = useAppSelector(
    (state) => state.trainInfo.trainInfo.status
  );

  const isRunning = trainStatus === "running";

  const dispatch = useAppDispatch();

  const [localOptimizerState, setLocalOptimizerState] =
    useState<Record<string, OptimizerData>>(optimizerStateServer);

  const handleOptimizerUpdateApply = (
    newState: Record<string, OptimizerData>,
    oldState: Record<string, OptimizerData>
  ) => {
    const paramsUpdated: Record<string, OptimizerData> = {};
    for (const key in newState) {
      if (newState[key].value !== oldState[key].value) {
        paramsUpdated[key] = newState[key];
      }
    }
    const trainCommand = generateOptimizerUpdateTrainCommand(paramsUpdated);
    dispatch(postTrainCommand(trainCommand));
  };

  useEffect(() => {
    // Initialize local state with server state
    setLocalOptimizerState(optimizerStateServer);
  }, [optimizerStateServer]);

  const displayControl = () => {
    return (
      <div className="optimizer-control-wrapper flex-1 overflow-auto px-2 py-2 flex-grow">
        <div className="optimizer-parameter-list space-y-2">
          {Object.entries(localOptimizerState).map(([key, param]) =>
            key == "optimizer_name" && typeof param.value === "string" ? (
              <KeyValueDisplay
                key={key}
                name="Optimizer Name"
                value={param.value}
                className="text-black p-2"
              />) :
            typeof param.value === "number" ? (
              <NumericalInputControlWithSlider
                key={key}
                id={key + "-input"}
                label={param.name}
                value={param.value}
                step={1e-7}
                min={0}
                max={1e-3}
                onChange={(newValue) => {
  
                  const updatedParams = {
                    ...localOptimizerState,
                    [key]: { ...param, value: newValue },
                  };
                  setLocalOptimizerState(updatedParams);
                }}
                className="text-black p-4 text-bold"
              />
            ) : typeof param.value === "boolean" ? (
              <BoolControlWithSwitch
                key={key}
                id={key + "-switch"}
                label={param.name}
                value={param.value}
                onChange={(newValue) => {
                  const updatedParams = {
                    ...localOptimizerState,
                    [key]: { ...param, value: newValue },
                  };
                  setLocalOptimizerState(updatedParams);
                }}
                className="text-black p-2"
              />
            ) : Array.isArray(param.value) ? (
              <FixedLengthNumericalListControl
                key={key}
                id={key + "-list"}
                label={param.name}
                value={param.value as number[]}
                onChange={(newValue) => {
                  const updatedParams = {
                    ...localOptimizerState,
                    [key]: { ...param, value: newValue },
                  };
                  setLocalOptimizerState(updatedParams);
                }}
                className="text-black p-2"
              />
            ) : (
              <Box
                key={key}
                sx={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 1,
                  width: "100%",
                }}
                className="w-full"
              >
                <Typography variant="subtitle2" fontWeight={400}>
                  {param.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {JSON.stringify(param.value)}
                </Typography>
              </Box>
            )
          )}
        </div>

        {/* Commit Button */}
        <button
          className="w-full bg-gray-600 text-white py-2 px-6 font-semibold hover:bg-gray-700 transition-colors duration-200 border-gray-300 mt-8"
          onClick={() => {
            handleOptimizerUpdateApply(
              localOptimizerState,
              optimizerStateServer
            );
          }}
        >
          Apply
        </button>
      </div>
    );
  };

  return (
    <div className={clsx("optimizer-control flex flex-col h-full", className)}>
      <h4 className="flex-shrink-0 flex items-center justify-start text-left text-lg font-semibold p-2 bg-gray-200">
        Optimizer Control Panel
      </h4>
      {isRunning ? (
        displayControl()
      ) : (
        <p className="p-4 text-sm text-gray-500">
          Optimizer control is disabled while training is running.
        </p>
      )}
    </div>
  );
};

export default OptimizerControl;
