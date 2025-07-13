import React, { useState } from "react";
import clsx from "clsx";
import { useAppSelector } from "../../hooks/userTypedHooks";
import type { ModelDataNode } from "../../features/modelInfo/type";
import { InfoDisplay, NumericalControl, ActionControl} from "../sharedControl/sharedControl";

export interface ControlSectionProps {
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
