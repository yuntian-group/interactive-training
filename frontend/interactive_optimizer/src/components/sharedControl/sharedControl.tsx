import React, { useState } from "react";
import clsx from "clsx";
import { Plus, X } from "lucide-react";

export interface InfoFieldProps {
    label: string;
    value: string;
}
const InfoField: React.FC<InfoFieldProps> = ({ label, value }) => (
    <div className="bg-gray-100 p-1.5">
        <div className="text-[10px] font-medium text-gray-500 text-left mb-1">
            {label}
        </div>
        <p className="text-xs font-semibold text-gray-900 truncate">{value}</p>
    </div>
);

export interface InfoDisplayProps {
    fields: Array<{ label: string; value: string }>;
}

export const InfoDisplay: React.FC<InfoDisplayProps> = ({ fields }) => (
    <div className="space-y-2">
        <div className="text-sm font-medium text-gray-700">Layer Information</div>
        <div className="grid grid-cols-4 gap-1">
            {fields.map((field, index) => (
                <InfoField key={index} label={field.label} value={field.value} />
            ))}
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
    const [inputValue, setInputValue] = useState(value.toString());

    const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setInputValue(e.target.value);
    };

    const handleInputBlur = () => {
        const numValue = parseFloat(inputValue);
        if (!isNaN(numValue) && numValue >= min && numValue <= max) {
            onChange(numValue);
        } else {
            setInputValue(value.toString());
        }
    };

    const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
        if (e.key === 'Enter') {
            handleInputBlur();
        }
    };

    React.useEffect(() => {
        setInputValue(value.toString());
    }, [value]);

    return (
        <div className="bg-gray-50 p-2 space-y-2">
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
    variant?: 'primary' | 'secondary' | 'danger';
    onClick: () => void;
}

export const ActionControl: React.FC<ActionControlProps> = ({ 
    label, 
    variant = 'primary',
    onClick 
}) => {
    const variantClasses = {
        primary: "bg-blue-600 text-white hover:bg-blue-700",
        secondary: "bg-gray-600 text-white hover:bg-gray-700",
        danger: "bg-red-600 text-white hover:bg-red-700"
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
    className
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
        if (e.key === 'Enter') {
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
}> = ({ label, value, onChange, id, min, max, className}) => (
    <InputControl
        label={label}
        value={value}
        onChange={onChange}
        id={id}
        parser={(input) => {
            const num = parseFloat(input);
            return isNaN(num) ? null : num;
        }}
        formatter={(val) => val.toString()}
        validator={(val) => {
            if (min !== undefined && val < min) return false;
            if (max !== undefined && val > max) return false;
            return true;
        }}
        className={className}
    />
);


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
    className
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
        if (e.key === 'Enter') {
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
        if (newKey.trim() && newKey !== oldKey && !value.hasOwnProperty(newKey.trim())) {
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
        if (e.key === 'Enter') {
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