import React, { useState } from "react";
import clsx from "clsx";
import { Plus, X, Play } from "lucide-react";
import type {
  ModelOperation,
  SingleSignature,
} from "../../features/modelInfo/type";

export interface InfoFieldProps {
  label: string;
  value: string;
  className?: string;
}
export const InfoField: React.FC<InfoFieldProps> = ({ label, value, className}) => (
  <div className={clsx("bg-gray-100 p-1.5", className)}>
    <div className="text-[10px] font-medium text-gray-500 text-left mb-1">
      {label}
    </div>
    <p className="text-xs font-semibold text-gray-900 truncate">{value}</p>
  </div>
);

export interface KeyValueDisplayProps {
  name: string;
  value: string | number | boolean;
  className?: string;
}

export const KeyValueDisplay: React.FC<KeyValueDisplayProps> = ({ name, value, className }) => (
  <div className={clsx("bg-gray-100 p-2", className)}>
    <div className="flex justify-start flex-col">
      <span className="text-xs font-medium text-gray-700">{name}</span>
      <span className="text-md font-mono text-gray-900 pt-2">
        {typeof value === 'boolean' ? (value ? 'true' : 'false') : String(value)}
      </span>
    </div>
  </div>
);

export interface NumericalControlProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  onApply: (value: number) => void;
  id: string;
}

export const NumericalControl: React.FC<NumericalControlProps> = ({
  label,
  value,
  min,
  max,
  step,
  onChange,
  onApply,
  id,
}) => {
  const [inputValue, setInputValue] = useState(value.toPrecision(3));

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    const numValue = parseFloat(inputValue);
    if (!isNaN(numValue) && numValue >= min && numValue <= max) {
      onChange(numValue);
    } else {
      setInputValue(value.toPrecision(3));
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleInputBlur();
    }
  };

  React.useEffect(() => {
    setInputValue(value.toPrecision(3));
  }, [value]);

  return (
    <div className="bg-gray-100 p-2 space-y-2">
      <div className="flex justify-between items-center">
        <label htmlFor={id} className="text-xs font-medium text-gray-700">
          {label}
        </label>
        <input
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onKeyPress={handleKeyPress}
          className="px-2 py-1 bg-white border border-gray-300 text-xs font-mono text-gray-900 w-20 text-center focus:outline-none focus:shadow-[0_0_0_1px_black] hover:shadow-[0_0_0_1px_black] transition-shadow"
        />
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-2 bg-gray-200 appearance-none cursor-pointer rounded-sm slider"
      />
      <button
        onClick={() => onApply(value)}
        className="w-full px-2 py-1.5 bg-blue-600 text-white text-xs font-medium hover:bg-blue-700 transition-colors rounded-sm"
      >
        Apply
      </button>
    </div>
  );
};

export interface ActionControlProps {
  label: string;
  variant?: "primary" | "secondary" | "danger";
  onClick: () => void;
}

export const ActionControl: React.FC<ActionControlProps> = ({
  label,
  variant = "primary",
  onClick,
}) => {
  const variantClasses = {
    primary: "bg-blue-600 text-white hover:bg-blue-700",
    secondary: "bg-gray-600 text-white hover:bg-gray-700",
    danger: "bg-red-600 text-white hover:bg-red-700",
  };

  return (
    <button
      className={clsx(
        "flex-1 px-2 py-1 text-xs transition-colors",
        variantClasses[variant]
      )}
      onClick={onClick}
    >
      {label}
    </button>
  );
};

export interface InputControlProps<T> {
  label: string;
  value: T;
  onChange: (value: T) => void;
  id: string;
  parser: (input: string) => T | null;
  formatter: (value: T) => string;
  validator?: (value: T) => boolean;
  className?: string;
}

export const InputControl = <T,>({
  label,
  value,
  onChange,
  id,
  parser,
  formatter,
  validator,
  className,
}: InputControlProps<T>) => {
  const [inputValue, setInputValue] = useState(formatter(value));

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    const parsedValue = parser(inputValue);
    if (parsedValue !== null && (!validator || validator(parsedValue))) {
      onChange(parsedValue);
    } else {
      setInputValue(formatter(value));
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleInputBlur();
    }
  };

  React.useEffect(() => {
    setInputValue(formatter(value));
  }, [value, formatter]);

  return (
    <div className={clsx("bg-gray-50 p-2 space-y-2", className)}>
      <div className="flex justify-between items-center">
        <label htmlFor={id} className="text-xs font-medium text-gray-700">
          {label}
        </label>
        <input
          id={id}
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onKeyPress={handleKeyPress}
          className="px-2 py-1 bg-white border border-gray-300 text-xs font-mono text-gray-900 w-20 text-center focus:outline-none focus:shadow-[0_0_0_1px_black] hover:shadow-[0_0_0_1px_black] transition-shadow"
        />
      </div>
    </div>
  );
};

export const NumericalInputControl: React.FC<{
  label: string;
  value: number;
  onChange: (value: number) => void;
  id: string;
  min?: number;
  max?: number;
  className?: string;
}> = ({ label, value, onChange, id, min, max, className }) => (
  <InputControl
    label={label}
    value={value}
    onChange={onChange}
    id={id}
    parser={(input) => {
      const num = parseFloat(input);
      return isNaN(num) ? null : num;
    }}
    formatter={(val) => val.toPrecision(3)}
    validator={(val) => {
      if (min !== undefined && val < min) return false;
      if (max !== undefined && val > max) return false;
      return true;
    }}
    className={className}
  />
);

export const NumericalInputControlWithSlider: React.FC<{
  label: string;
  value: number;
  onChange: (value: number) => void;
  id: string;
  min: number;
  max: number;
  step: number;
  className?: string;
  onApply?: (value: number) => void;
}> = ({ label, value, onChange, id, min, max, step, className, onApply }) => {
  const [inputValue, setInputValue] = useState(value.toPrecision(3));

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    const numValue = parseFloat(inputValue);
    if (!isNaN(numValue) && numValue >= min && numValue <= max) {
      onChange(numValue);
    } else {
      setInputValue(value.toPrecision(3));
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      handleInputBlur();
    }
  };

  React.useEffect(() => {
    setInputValue(value.toPrecision(3));
  }, [value]);

  return (
    <div className={clsx("bg-gray-100 p-2 space-y-1", className)}>
      <style>{`
                .slider-thumb-black-square::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    appearance: none;
                    width: 10px;
                    height: 10px;
                    background: black;
                    cursor: pointer;
                    border-radius: 0;
                }

                .slider-thumb-black-square::-moz-range-thumb {
                    width: 10px;
                    height: 10px;
                    background: black;
                    cursor: pointer;
                    border-radius: 0;
                }
            `}</style>
      <div className="flex justify-between items-center">
        <label
          htmlFor={id + "-input"}
          className="text-xs font-medium text-gray-700"
        >
          {label}
        </label>
        <input
          id={id + "-input"}
          type="text"
          value={inputValue}
          onChange={handleInputChange}
          onBlur={handleInputBlur}
          onKeyPress={handleKeyPress}
          className="px-2 py-1 bg-white border border-gray-300 text-xs font-mono text-gray-900 w-20 text-center focus:outline-none focus:shadow-[0_0_0_1px_black] hover:shadow-[0_0_0_1px_black] transition-shadow"
        />
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1 bg-gray-200 appearance-none cursor-pointer slider slider-thumb-black-square"
      />
      {onApply && (
        <div className="flex justify-end">
          <button
            onClick={() => onApply(value)}
            className="px-2 py-1.5 bg-gray-600 text-white text-xs font-medium hover:bg-gray-700 transition-colors"
          >
            Apply
          </button>
        </div>
      )}
    </div>
  );
};

export interface BoolControlWithSwitchProps {
  label: string;
  value: boolean;
  onChange: (value: boolean) => void;
  id: string;
  className?: string;
}

export const BoolControlWithSwitch: React.FC<BoolControlWithSwitchProps> = ({
  label,
  value,
  onChange,
  id,
  className,
}) => {
  return (
    <div className={clsx("bg-gray-100 p-2", className)}>
      <div className="flex justify-between items-center">
        <span className="text-xs font-medium text-gray-700">{label}</span>
        <label htmlFor={id} className="flex items-center cursor-pointer">
          <div className="relative">
            <input
              id={id}
              type="checkbox"
              className="sr-only"
              checked={value}
              onChange={() => onChange(!value)}
            />
            <div className="block bg-gray-200 w-10 h-6"></div>
            <div
              className={clsx(
                "absolute left-1 top-1 w-4 h-4 transition-transform",
                value ? "transform translate-x-full bg-black" : "bg-white"
              )}
            ></div>
          </div>
        </label>
      </div>
    </div>
  );
};

export interface ListInputControlProps {
  label: string;
  value: string[];
  onChange: (value: string[]) => void;
  id: string;
  className?: string;
}

export const ListInputControl: React.FC<ListInputControlProps> = ({
  label,
  value,
  onChange,
  id,
  className,
}) => {
  const [newItem, setNewItem] = useState("");

  const addItem = () => {
    if (newItem.trim() && !value.includes(newItem.trim())) {
      onChange([...value, newItem.trim()]);
      setNewItem("");
    }
  };

  const removeItem = (index: number) => {
    onChange(value.filter((_, i) => i !== index));
  };

  const updateItem = (index: number, newValue: string) => {
    const updatedItems = [...value];
    updatedItems[index] = newValue;
    onChange(updatedItems);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      addItem();
    }
  };

  return (
    <div className={clsx("bg-gray-50 px-2 space-y-2", className)}>
      <label htmlFor={id} className="text-xs font-medium text-gray-700 block">
        {label}
      </label>

      {value.length > 0 && (
        <div className="overflow-y-auto px-2 py-1 gap-2">
          {value.map((item, index) => (
            <div key={index} className="flex items-center space-x-2">
              <input
                type="text"
                value={item}
                onChange={(e) => updateItem(index, e.target.value)}
                className="flex-1 px-2 py-1 bg-white border border-gray-300 text-xs font-mono  text-gray-900 focus:outline-none focus:shadow-[0_0_0_1px_black] hover:shadow-[0_0_0_1px_black] transition-shadow"
              />
              <button
                onClick={() => removeItem(index)}
                className="px-1 py-1 bg-transparent border border-gray-300 text-xs text-gray-600 hover:bg-gray-100 hover:text-gray-800 transition-colors flex items-center justify-center"
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="flex px-2 py-1 gap-2">
        <input
          id={id}
          type="text"
          value={newItem}
          onChange={(e) => setNewItem(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Add new item..."
          className="flex-1 px-2 py-1 bg-white border border-gray-300 text-xs text-gray-900 focus:outline-none focus:shadow-[0_0_0_1px_black] hover:shadow-[0_0_0_1px_black] transition-shadow"
        />
        <button
          onClick={addItem}
          className="px-1 py-1 bg-transparent border border-gray-300 text-xs text-gray-800 hover:bg-gray-100 hover:text-black transition-colors flex items-center justify-center"
        >
          <Plus size={12} />
        </button>
      </div>
    </div>
  );
};

export interface DictInputControlProps {
  label: string;
  value: Record<string, string>;
  onChange: (value: Record<string, string>) => void;
  id: string;
}

export const DictInputControl: React.FC<DictInputControlProps> = ({
  label,
  value,
  onChange,
  id,
}) => {
  const [newKey, setNewKey] = useState("");
  const [newValue, setNewValue] = useState("");

  const addEntry = () => {
    if (newKey.trim() && !value.hasOwnProperty(newKey.trim())) {
      onChange({ ...value, [newKey.trim()]: newValue.trim() });
      setNewKey("");
      setNewValue("");
    }
  };

  const removeEntry = (key: string) => {
    const updatedDict = { ...value };
    delete updatedDict[key];
    onChange(updatedDict);
  };

  const updateKey = (oldKey: string, newKey: string) => {
    if (
      newKey.trim() &&
      newKey !== oldKey &&
      !value.hasOwnProperty(newKey.trim())
    ) {
      const updatedDict = { ...value };
      updatedDict[newKey.trim()] = updatedDict[oldKey];
      delete updatedDict[oldKey];
      onChange(updatedDict);
    }
  };

  const updateValue = (key: string, newValue: string) => {
    onChange({ ...value, [key]: newValue });
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      addEntry();
    }
  };

  return (
    <div className="bg-gray-50 p-2 space-y-2">
      <label htmlFor={id} className="text-xs font-medium text-gray-700 block">
        {label}
      </label>

      <div className="flex gap-2">
        <input
          id={id}
          type="text"
          value={newKey}
          onChange={(e) => setNewKey(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Key..."
          className="flex-1 px-2 py-1 bg-white border border-gray-300 text-xs text-gray-900 focus:outline-none focus:shadow-[0_0_0_1px_black] hover:shadow-[0_0_0_1px_black] transition-shadow"
        />
        <input
          type="text"
          value={newValue}
          onChange={(e) => setNewValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Value..."
          className="flex-1 px-2 py-1 bg-white border border-gray-300 text-xs text-gray-900 focus:outline-none focus:shadow-[0_0_0_1px_black] hover:shadow-[0_0_0_1px_black] transition-shadow"
        />
        <button
          onClick={addEntry}
          className="px-1 py-1 bg-transparent border border-gray-300 text-xs text-gray-800 hover:bg-gray-100 hover:text-black transition-colors flex items-center justify-center"
        >
          <Plus size={12} />
        </button>
      </div>

      {Object.keys(value).length > 0 && (
        <div className="space-y-1.5 max-h-32 overflow-y-auto">
          {Object.entries(value).map(([key, val], index) => (
            <div key={index} className="flex gap-2 items-center">
              <input
                type="text"
                value={key}
                onChange={(e) => updateKey(key, e.target.value)}
                className="flex-1 px-2 py-1 bg-white border border-gray-300 text-xs font-mono text-gray-900 focus:outline-none focus:shadow-[0_0_0_1px_black] hover:shadow-[0_0_0_1px_black] transition-shadow"
              />
              <span className="text-xs text-gray-500 font-medium">:</span>
              <input
                type="text"
                value={val}
                onChange={(e) => updateValue(key, e.target.value)}
                className="flex-1 px-2 py-1 bg-white border border-gray-300 text-xs font-mono text-gray-900 focus:outline-none focus:shadow-[0_0_0_1px_black] hover:shadow-[0_0_0_1px_black] transition-shadow"
              />
              <button
                onClick={() => removeEntry(key)}
                className="px-1 py-1 bg-transparent border border-gray-300 text-xs text-gray-600 hover:bg-gray-100 hover:text-gray-800 transition-colors flex items-center justify-center"
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export interface FixedLengthNumericalListControlProps {
  label: string;
  value: number[];
  onChange: (value: number[]) => void;
  id: string;
  min?: number;
  max?: number;
  step?: number;
  precision?: number;
  className?: string;
}

export const FixedLengthNumericalListControl: React.FC<
  FixedLengthNumericalListControlProps
> = ({
  label,
  value,
  onChange,
  id,
  min,
  max,
  step = 0.01,
  precision = 3,
  className,
}) => {
  const [inputValues, setInputValues] = useState(
    value.map((v) => v.toPrecision(precision))
  );

  const updateItem = (index: number, inputValue: string) => {
    const newInputValues = [...inputValues];
    newInputValues[index] = inputValue;
    setInputValues(newInputValues);
  };

  const handleInputBlur = (index: number) => {
    const numValue = parseFloat(inputValues[index]);
    if (!isNaN(numValue)) {
      const clampedValue =
        min !== undefined && max !== undefined
          ? Math.max(min, Math.min(max, numValue))
          : numValue;

      const newValue = [...value];
      newValue[index] = clampedValue;
      onChange(newValue);

      // Update input display with the actual value
      const newInputValues = [...inputValues];
      newInputValues[index] = clampedValue.toPrecision(precision);
      setInputValues(newInputValues);
    } else {
      // Reset to current value if invalid
      const newInputValues = [...inputValues];
      newInputValues[index] = value[index].toPrecision(precision);
      setInputValues(newInputValues);
    }
  };

  const handleKeyPress = (
    e: React.KeyboardEvent<HTMLInputElement>,
    index: number
  ) => {
    if (e.key === "Enter") {
      handleInputBlur(index);
    }
  };

  React.useEffect(() => {
    setInputValues(value.map((v) => v.toPrecision(precision)));
  }, [value, precision]);

  return (
    <div className={clsx("bg-gray-100 p-2 space-y-2", className)}>
      <label htmlFor={id} className="text-xs font-medium text-gray-700 block">
        {label} ({value.length} items)
      </label>

      <div className="space-y-1.5 max-h-32 overflow-y-auto p-1">
        {value.map((_item, index) => (
          <div key={index} className="flex items-center space-x-2">
            <span className="text-xs text-gray-500 font-medium w-4">
              {index}:
            </span>
            <input
              type="text"
              value={inputValues[index]}
              onChange={(e) => updateItem(index, e.target.value)}
              onBlur={() => handleInputBlur(index)}
              onKeyPress={(e) => handleKeyPress(e, index)}
              className="flex-1 px-2 py-1 bg-white border border-gray-300 text-xs font-mono text-gray-900 focus:outline-none focus:shadow-[0_0_0_1px_black] hover:shadow-[0_0_0_1px_black] transition-shadow"
            />
            {step && (
              <div className="flex space-x-1">
                <button
                  onClick={() => {
                    const newValue = [...value];
                    const decremented = newValue[index] - step;
                    newValue[index] =
                      min !== undefined
                        ? Math.max(min, decremented)
                        : decremented;
                    onChange(newValue);
                  }}
                  className="px-1 py-1 bg-transparent border border-gray-300 text-xs text-gray-600 hover:bg-gray-100 hover:text-gray-800 transition-colors"
                >
                  -
                </button>
                <button
                  onClick={() => {
                    const newValue = [...value];
                    const incremented = newValue[index] + step;
                    newValue[index] =
                      max !== undefined
                        ? Math.min(max, incremented)
                        : incremented;
                    onChange(newValue);
                  }}
                  className="px-1 py-1 bg-transparent border border-gray-300 text-xs text-gray-600 hover:bg-gray-100 hover:text-gray-800 transition-colors"
                >
                  +
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {(min !== undefined || max !== undefined) && (
        <div className="text-xs text-gray-500">
          Range: {min !== undefined ? min : "-∞"} to{" "}
          {max !== undefined ? max : "∞"}
        </div>
      )}
    </div>
  );
};

export interface RemoteFunctionControlProps {
  functionInfo: ModelOperation;
  onApply: (parameters: Record<string, any>) => void;
  className?: string;
  loading?: boolean;
}

export const RemoteFunctionControl: React.FC<RemoteFunctionControlProps> = ({
  functionInfo,
  onApply,
  className,
  loading = false,
}) => {
  const [parameters, setParameters] = useState<Record<string, any>>(() => {
    const initialParams: Record<string, any> = {};
    functionInfo.signature.forEach((param) => {
      initialParams[param.name] = param.value;
    });
    return initialParams;
  });

  const updateParameter = (name: string, value: any) => {
    setParameters((prev) => ({ ...prev, [name]: value }));
  };

  const handleCall = () => {
    onApply(parameters);
  };

  const renderParameterControl = (param: SingleSignature) => {
    const baseId = `${functionInfo.name}-${param.name}`;

    switch (param.type.toLowerCase()) {
      case "bool":
      case "boolean":
        return (
          <BoolControlWithSwitch
            key={param.name}
            id={baseId}
            label={param.name}
            value={parameters[param.name] as boolean}
            onChange={(value) => updateParameter(param.name, value)}
          />
        );

      case "int":
      case "integer":
        if (param.min !== undefined && param.max !== undefined) {
          return (
            <NumericalInputControlWithSlider
              key={param.name}
              id={baseId}
              label={param.name}
              value={parameters[param.name] as number}
              onChange={(value) =>
                updateParameter(param.name, Math.round(value))
              }
              min={param.min}
              max={param.max}
              step={param.step || 1}
            />
          );
        } else {
          return (
            <NumericalInputControl
              key={param.name}
              id={baseId}
              label={param.name}
              value={parameters[param.name] as number}
              onChange={(value) =>
                updateParameter(param.name, Math.round(value))
              }
              min={param.min}
              max={param.max}
            />
          );
        }

      case "float":
      case "number":
        if (param.min !== undefined && param.max !== undefined) {
          return (
            <NumericalInputControlWithSlider
              key={param.name}
              id={baseId}
              label={param.name}
              value={parameters[param.name] as number}
              onChange={(value) => updateParameter(param.name, value)}
              min={param.min}
              max={param.max}
              step={param.step || 0.01}
            />
          );
        } else {
          return (
            <NumericalInputControl
              key={param.name}
              id={baseId}
              label={param.name}
              value={parameters[param.name] as number}
              onChange={(value) => updateParameter(param.name, value)}
              min={param.min}
              max={param.max}
            />
          );
        }

      case "str":
      case "string":
        return (
          <InputControl
            key={param.name}
            id={baseId}
            label={param.name}
            value={parameters[param.name] as string}
            onChange={(value) => updateParameter(param.name, value)}
            parser={(input) => input}
            formatter={(val) => val}
          />
        );

      case "list":
      case "array":
        if (
          Array.isArray(parameters[param.name]) &&
          parameters[param.name].length > 0 &&
          typeof parameters[param.name][0] === "number"
        ) {
          return (
            <FixedLengthNumericalListControl
              key={param.name}
              id={baseId}
              label={param.name}
              value={parameters[param.name] as number[]}
              onChange={(value) => updateParameter(param.name, value)}
              min={param.min}
              max={param.max}
              step={param.step}
            />
          );
        } else {
          return (
            <ListInputControl
              key={param.name}
              id={baseId}
              label={param.name}
              value={parameters[param.name] as string[]}
              onChange={(value) => updateParameter(param.name, value)}
            />
          );
        }

      case "dict":
      case "object":
        return (
          <DictInputControl
            key={param.name}
            id={baseId}
            label={param.name}
            value={parameters[param.name] as Record<string, string>}
            onChange={(value) => updateParameter(param.name, value)}
          />
        );

      default:
        return (
          <InputControl
            key={param.name}
            id={baseId}
            label={`${param.name} (${param.type})`}
            value={parameters[param.name]}
            onChange={(value) => updateParameter(param.name, value)}
            parser={(input) => {
              try {
                return JSON.parse(input);
              } catch {
                return input;
              }
            }}
            formatter={(val) =>
              typeof val === "string" ? val : JSON.stringify(val)
            }
          />
        );
    }
  };

  React.useEffect(() => {
    const newParameters: Record<string, any> = {};
    functionInfo.signature.forEach((param) => {
      newParameters[param.name] = param.value;
    });
    setParameters(newParameters);
  }, [functionInfo]);

  return (
    <div
      className={clsx("border border-gray-200 space-y-1", className)}
    >
      <div className="bg-gray-50 px-3 py-2">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800">
            {functionInfo.name}()
          </h3>
          <button
            onClick={handleCall}
            disabled={loading}
            className={clsx(
              "px-2 py-1 text-xs font-medium transition-colors rounded-sm flex items-center gap-1",
              loading
                ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                : "bg-gray-600 text-white hover:bg-gray-700"
            )}
          >
            {loading ? (
              "..."
            ) : (
              <>
                <Play size={10} fill="currentColor" />
                Call
              </>
            )}
          </button>
        </div>
        {functionInfo.signature.length === 0 && (
          <p className="text-xs text-gray-500 mt-1">No parameters</p>
        )}
      </div>

      {functionInfo.signature.length > 0 && (
        <div className="space-y-1">
          {functionInfo.signature.map(renderParameterControl)}
        </div>
      )}
    </div>
  );
};
