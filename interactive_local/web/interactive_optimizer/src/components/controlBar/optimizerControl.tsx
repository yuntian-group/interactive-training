import { Divider } from "@mui/material";
import clsx from "clsx";
import React, { useState, useEffect } from "react";
import { Box, Slider, Typography, TextField } from "@mui/material";

interface OptimizerParams {
  learningRate: number;
}

const OptimizerParameterControl: React.FC<{
  label: string;
  value: number;
  step: number;
  min?: number;
  max?: number;
  onChange: (value: number) => void;
  className?: string;
}> = ({ label, value, step, min = 0, max = 0.1, onChange, className }) => {
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
  const [params, setParams] = useState<OptimizerParams>({
    learningRate: 1e-5, // Default learning rate
  });

  const handleInputChange = (
    key: keyof OptimizerParams,
    value: string | number
  ) => {
    setParams((prev) => ({
      ...prev,
      [key]: typeof value === "string" ? Number(value) : value,
    }));
  };

  const handleCommit = () => {
    console.log("Committing optimizer parameters:", params);
    // Apply the parameters here
  };

  return (
    <div
      className={clsx(
        "optimizer-control bg-white text-back p-4 border border-gray-300",
        className
      )}
    >
      <h2 className="text-xl font-bold mb-4">Optimizer Control Panel</h2>
      <Divider />
      <div className="gap-4 mb-6 mt-4">
        {/* Learning Rate Slider */}
        <OptimizerParameterControl
          label="Learning Rate"
          value={params.learningRate}
          step={1e-8}
          min={1e-8}
          max={1e-3}
          onChange={(value) => handleInputChange("learningRate", value)}
        />
      </div>

      {/* Commit Button */}
      <button
        onClick={handleCommit}
        className="w-full bg-gray-100 text-black py-2 px-6 font-semibold hover:bg-gray-200 transition-colors duration-200 border-gray-300"
      >
        Apply
      </button>
    </div>
  );
};

export default OptimizerControl;
