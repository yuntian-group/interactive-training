import clsx from "clsx";
import React, {useState } from "react";
import SideControl from "./sideControl";
import OptimizerControl from "./optimizerControl";
import CheckpointControl from "./checkPointControl";
import TrainInfoDisplay from "./trainInfoDisplay";
import ModelInfoDisplay from "./modelInfoDisplay";

type Props = React.HTMLAttributes<HTMLDivElement>;

const ControlBar: React.FC<Props> = ({ className }: Props) => {

    const [activePanel, setActivePanel] = useState<string>("Info");

    return (
        <div className={clsx("control-bar bg-white flex flex-row border-r-1 border-gray-400 flex-shrink-0 overflow-auto", className)}>
            <SideControl className="w-16 h-full flex-shrink-0" onSideItemClick={(label: string) => {
                setActivePanel(label);
            }} initActiveItem="Info" />
            {/* Render the active panel based on the selected item */}
            {activePanel === "Info" &&  <TrainInfoDisplay className="flex-1 flex-col w-full h-full flex-shrink-0 overflow-auto" />}
            {activePanel === "Optimizer" && <OptimizerControl className="flex-1 flex-col w-full h-full flex-shrink-0 overflow-auto" />}
            {activePanel === "Checkpoint" && <CheckpointControl className="flex-1 flex-col w-full h-full flex-shrink-0 overflow-auto" />}
            {activePanel === "Dataset" && <h4>Data Work In Progress</h4>}
            {activePanel === "Model" && <ModelInfoDisplay className="flex-1 flex-col w-full h-full flex-shrink-0 overflow-auto" />}
            {/* Add more panels as needed */}
        </div>
    );
};

export default ControlBar;