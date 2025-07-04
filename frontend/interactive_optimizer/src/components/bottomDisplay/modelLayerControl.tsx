// /root/interactive_trainer/interactive_local/web/interactive_optimizer/src/components/bottomDisplay/modelLayerControl.tsx
import React, { useState } from "react";
import clsx from "clsx";
import { useAppSelector } from "../../hooks/userTypedHooks";
import type { ModelDataNode } from "../../features/modelInfo/type";

interface InfoFieldProps {
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

interface InfoDisplayProps {
    fields: Array<{ label: string; value: string }>;
}

const InfoDisplay: React.FC<InfoDisplayProps> = ({ fields }) => (
    <div className="space-y-2">
        <div className="text-sm font-medium text-gray-700">Layer Information</div>
        <div className="grid grid-cols-4 gap-1">
            {fields.map((field, index) => (
                <InfoField key={index} label={field.label} value={field.value} />
            ))}
        </div>
    </div>
);

interface NumericalControlProps {
    label: string;
    value: number;
    min: number;
    max: number;
    step: number;
    onChange: (value: number) => void;
    onApply: (value: number) => void;
    id: string;
}

const NumericalControl: React.FC<NumericalControlProps> = ({
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
                    className="px-1 py-0.5 bg-gray-200 text-xs font-mono text-gray-700 w-16 text-center rounded-none focus:outline-none focus:ring-2 focus:ring-black-500"
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
                className="w-full h-1.5 bg-gray-200 appearance-none cursor-pointer rounded-none"
            />
            <button
                onClick={() => onApply(value)}
                className="w-full px-2 py-1 bg-blue-600 text-white text-xs hover:bg-blue-700 transition-colors"
            >
                Apply
            </button>
        </div>
    );
};

interface ActionControlProps {
    label: string;
    variant?: 'primary' | 'secondary' | 'danger';
    onClick: () => void;
}

const ActionControl: React.FC<ActionControlProps> = ({ 
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

interface ControlSectionProps {
    numericalControls?: Array<{
        id: string;
        label: string;
        value: number;
        min: number;
        max: number;
        step: number;
        onChange: (value: number) => void;
        onApply: (value: number) => void;
    }>;
    actionControls?: Array<{
        label: string;
        variant?: 'primary' | 'secondary' | 'danger';
        onClick: () => void;
    }>;
}

const ControlSection: React.FC<ControlSectionProps> = ({ 
    numericalControls = [], 
    actionControls = [] 
}) => (
    <div className="space-y-2">
        <h3 className="text-sm font-medium text-gray-700">Controls</h3>

        <div className="numerical-control-section grid grid-cols-2 gap-2">
            {/* Numerical Controls */}
            {numericalControls.length > 0 && (
                    numericalControls.map((control) => (
                        <NumericalControl
                            key={control.id}
                            {...control}
                        />
                    ))

            )}
        </div>
        
        {/* Action Controls */}
        {actionControls.length > 0 && (
            <div className="flex gap-1 bg-gray-100 p-2">
                {actionControls.map((action, index) => (
                    <ActionControl
                        key={index}
                        {...action}
                    />
                ))}
            </div>
        )}
    </div>
);

interface ModelLayerControlProps {
    className: string;
    infoFields?: Array<{ label: string; value: string }>;
    numericalControls?: Array<{
        id: string;
        label: string;
        value: number;
        min: number;
        max: number;
        step: number;
        onChange: (value: number) => void;
        onApply: (value: number) => void;
    }>;
    actionControls?: Array<{
        label: string;
        variant?: 'primary' | 'secondary' | 'danger';
        onClick: () => void;
    }>;
}

const ModelLayerControl: React.FC<ModelLayerControlProps> = ({
    className,
    numericalControls,
    actionControls
}) => {
    const [dropoutRate, setDropoutRate] = useState(0.01);
    const [weightDecay, setWeightDecay] = useState(0.0001);
    const selectedLayer = useAppSelector((state) => state.modelInfo.selectedLayer);
    const layerInfo = useAppSelector((state) => state.modelInfo.nodeMap[selectedLayer] || {
        name: selectedLayer,
        moduleType: "Unknown",
        children: [],
        operators: [],
    } as ModelDataNode);

    const infoFields = [
        { label: "Layer Name", value: layerInfo.name || "Unknown" },
        { label: "Layer Type", value: layerInfo.moduleType || "Unknown" },
    ]

    // Default numerical controls if none provided
    const defaultNumericalControls = [
        {
            id: "dropout-slider",
            label: "Dropout Rate",
            value: dropoutRate,
            min: 0,
            max: 1,
            step: 0.01,
            onChange: setDropoutRate,
            onApply: (value: number) => console.log('Apply dropout:', value)
        },
        {
            id: "weight-decay-slider",
            label: "Gradient Norm",
            value: weightDecay,
            min: 0,
            max: 0.01,
            step: 0.0001,
            onChange: setWeightDecay,
            onApply: (value: number) => console.log('Apply weight decay:', value)
        }
    ];

    // Default action controls if none provided
    const defaultActionControls = [
        {
            label: "Freeze",
            variant: 'primary' as const,
            onClick: () => console.log('Freeze layer')
        },
        {
            label: "Re-init",
            variant: 'danger' as const,
            onClick: () => console.log('Re-initialize layer')
        }
    ];

    return (
        <div
            className={clsx(
                "modellayer p-3 bg-white border border-gray-200 shadow-sm flex flex-col space-y-4 flex-shrink-0 min-w-0",
                className
            )}
        >
            <InfoDisplay fields={infoFields} />
            <ControlSection 
                numericalControls={numericalControls || defaultNumericalControls}
                actionControls={actionControls || defaultActionControls}
            />
        </div>
    );
};

export default ModelLayerControl;
