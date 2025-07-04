import clsx from "clsx";
import { v4 as uuidv4 } from "uuid";
import React, { useState, useEffect } from "react";
import { useAppDispatch, useAppSelector } from "../../hooks/userTypedHooks";
import { Box, Slider, Typography, TextField } from "@mui/material";
import type { OptimizerData } from "../../features/optimizerState/type";
import type { TrainCommandData } from "../../features/trainCommand/types";
import { postTrainCommand } from "../../features/trainCommand/actions";

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

const OptimizerParameterControl: React.FC<{
  label: string;
  value: number;
  step?: number;
  min?: number;
  max?: number;
  onChange: (value: number) => void;
  className?: string;
}> = ({
  label,
  value,
  step = 1e-7,
  min = 0,
  max = 0.1,
  onChange,
  className,
}) => {
  const [tempValue, setTempValue] = useState<string>(value.toExponential(3));

  useEffect(() => {
    setTempValue(value.toExponential(3));
  }, [value]);

  const commitValue = () => {
    const parsed = Number(tempValue);
    if (!isNaN(parsed) && parsed >= min && parsed <= max) {
      onChange(parsed);
    } else {
      // revert to last valid
      setTempValue(value.toExponential(3));
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 1,
        width: "100%",
      }}
      className={clsx("optimizer-parameter-control", className)}
    >
      <Typography variant="subtitle2" fontWeight={400} sx={{ width: "100%" }}>
        {label}
      </Typography>
      <div className="flex flex-row w-full items-center justify-between">
        <Slider
          value={value}
          step={step}
          min={min}
          max={max}
          onChange={(_, v) => onChange(Array.isArray(v) ? v[0] : v)}
          valueLabelDisplay="auto"
          sx={{
            width: "60%",
            color: "#000",
            height: 6,
            marginRight: "10%",
            "& .MuiSlider-rail": { bgcolor: "#f0f0f0", opacity: 1 },
            "& .MuiSlider-track": { bgcolor: "#000" },
            "& .MuiSlider-thumb": {
              width: 16,
              height: 16,
              bgcolor: "#fff",
              border: "2px solid #000",
              "&:hover, &.Mui-focusVisible": { boxShadow: "none" },
            },
            "& .MuiSlider-valueLabel": {
              top: -28,
              bgcolor: "#000",
              color: "#fff",
              fontSize: 12,
              fontWeight: 600,
              borderRadius: 2,
            },
          }}
        />

        <TextField
          type="text"
          value={tempValue}
          onChange={(e) => setTempValue(e.target.value)}
          onBlur={commitValue}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              commitValue();
              e.currentTarget.blur();
            }
          }}
          inputProps={{
            inputMode: "decimal",
            pattern: "[0-9eE.\\+\\-]+",
            style: { textAlign: "center" },
          }}
          variant="outlined"
          size="small"
          sx={{
            width: "30%",
            // remove up/down arrows if the browser still shows them
            "& input::-webkit-outer-spin-button, & input::-webkit-inner-spin-button":
              {
                WebkitAppearance: "none",
                margin: 0,
              },
            "& input[type=number]": {
              MozAppearance: "textfield",
            },
            "& .MuiOutlinedInput-root": {
              bgcolor: "#fff",
              borderRadius: 0,
              fontSize: "12px",
              justifyContent: "left",
              "& fieldset": { borderColor: "#000" },
              "&:hover fieldset": { borderColor: "#000" },
              "&.Mui-focused fieldset": { borderColor: "#000" },
            },
          }}
        />
      </div>
    </Box>
  );
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


  const displayControl = () => {
    return (
      <div className="optimizer-control-wrapper flex-1 overflow-auto p-4 m-4 flex-grow">
        <div className="optimizer-parameter-list space-y-4 mt-4 mb-4">
          {Object.entries(localOptimizerState).map(([key, param]) => (
            <OptimizerParameterControl
              key={key}
              label={param.name}
              value={param.value}
              onChange={(value) => {
                console.log(`Updating ${param.name} to ${value}`);
                const updatedParams = {
                  ...localOptimizerState,
                  [key]: { ...param, value },
                };
                setLocalOptimizerState(updatedParams);
              }}
              className="w-full"
            />
          ))}
        </div>

        {/* Commit Button */}
        <button
          className="w-full bg-gray-100 text-black py-2 px-6 font-semibold hover:bg-gray-200 transition-colors duration-200 border-gray-300"
          onClick={() => {
            console.log("Committing optimizer state:", localOptimizerState);
            handleOptimizerUpdateApply(
              localOptimizerState,
              optimizerStateServer
            );
          }}
        >
          Apply
        </button>
      </div>
    )
  }

  return (
    <div className={clsx("optimizer-control flex flex-col h-full", className)}>
      <h4 className="flex-shrink-0 flex items-center justify-start text-left text-lg font-semibold p-2 bg-gray-200">
        Optimizer Control Panel
      </h4>
      {
        isRunning ? (
          displayControl()
        ) : (
            <p className="p-4 text-sm text-gray-500">Optimizer control is disabled while training is running.</p>
        )
      }
      
    </div>
  );
};

export default OptimizerControl;
