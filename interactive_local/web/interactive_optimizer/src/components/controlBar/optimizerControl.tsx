import clsx from "clsx";
import React, { useState } from "react";

interface OptimizerParams {
    learningRate: number;
    momentum: number;
    weightDecay: number;
    beta1: number;
    beta2: number;
    epsilon: number;
}

type Props = React.HTMLAttributes<HTMLDivElement>

const OptimizerControl: React.FC<Props> = ({ className }: Props) => {
    const [params, setParams] = useState<OptimizerParams>({
        learningRate: 0.001,
        momentum: 0.9,
        weightDecay: 0.0001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    });

    const handleInputChange = (key: keyof OptimizerParams, value: string | number) => {
        setParams(prev => ({
            ...prev,
            [key]: typeof value === 'string' ? value : Number(value)
        }));
    };

    const handleCommit = () => {
        console.log("Committing optimizer parameters:", params);
        // Apply the parameters here
    };

    return (
        <div className={clsx("optimizer-control bg-white text-back p-6 border border-gray-300", className)}>
            <h2 className="text-xl font-bold mb-4 text-center">Optimizer Control Panel</h2>
            
            <div className="grid grid-cols-2 gap-4 mb-6">

                {/* Learning Rate */}
                <div className="flex flex-col">
                    <label className="text-sm font-medium mb-1">Learning Rate</label>
                    <input
                        type="number"
                        step="0.0001"
                        value={params.learningRate}
                        onChange={(e) => handleInputChange('learningRate', e.target.value)}
                        className="bg-white text-black px-3 py-2 rounded border focus:outline-none focus:ring-2 focus:ring-gray-400"
                    />
                </div>

                {/* Momentum */}
                <div className="flex flex-col">
                    <label className="text-sm font-medium mb-1">Momentum</label>
                    <input
                        type="number"
                        step="0.01"
                        value={params.momentum}
                        onChange={(e) => handleInputChange('momentum', e.target.value)}
                        className="bg-white text-black px-3 py-2 border focus:outline-none focus:ring-2 focus:ring-gray-400"
                    />
                </div>

                {/* Weight Decay */}
                <div className="flex flex-col">
                    <label className="text-sm font-medium mb-1">Weight Decay</label>
                    <input
                        type="number"
                        step="0.0001"
                        value={params.weightDecay}
                        onChange={(e) => handleInputChange('weightDecay', e.target.value)}
                        className="bg-white text-black px-3 py-2 border focus:outline-none focus:ring-2 focus:ring-gray-400"
                    />
                </div>

                {/* Beta1 */}
                <div className="flex flex-col">
                    <label className="text-sm font-medium mb-1">Beta1</label>
                    <input
                        type="number"
                        step="0.01"
                        value={params.beta1}
                        onChange={(e) => handleInputChange('beta1', e.target.value)}
                        className="bg-white text-black px-3 py-2 border focus:outline-none focus:ring-2 focus:ring-gray-400"
                    />
                </div>

                {/* Beta2 */}
                <div className="flex flex-col">
                    <label className="text-sm font-medium mb-1">Beta2</label>
                    <input
                        type="number"
                        step="0.001"
                        value={params.beta2}
                        onChange={(e) => handleInputChange('beta2', e.target.value)}
                        className="bg-white text-black px-3 py-2 border focus:outline-none focus:ring-2 focus:ring-gray-400"
                    />
                </div>

                {/* Epsilon */}
                <div className="flex flex-col">
                    <label className="text-sm font-medium mb-1">Epsilon</label>
                    <input
                        type="number"
                        step="1e-9"
                        value={params.epsilon}
                        onChange={(e) => handleInputChange('epsilon', e.target.value)}
                        className="bg-white text-black px-3 py-2 border focus:outline-none focus:ring-2 focus:ring-gray-400"
                    />
                </div>
            </div>

            {/* Commit Button */}
            <button
                onClick={handleCommit}
                className="w-full bg-white text-black py-3 px-6 font-semibold hover:bg-gray-200 transition-colors duration-200 border-2 border-white"
            >
                Apply
            </button>
        </div>
    );
}

export default OptimizerControl;