import clsx from "clsx";
import React, {useState } from "react";
import SideControl from "./sideControl";
import OptimizerControl from "./optimizerControl";
import CheckpointControl from "./checkPointControl";
import TrainInfoDisplay from "./trainInfoDisplay";

type Props = React.HTMLAttributes<HTMLDivElement>;

const ControlBar: React.FC<Props> = ({ className }: Props) => {

    const [activePanel, setActivePanel] = useState<string>("Info");

    return (
        <div className={clsx("control-bar bg-white flex flex-row border-r-1 border-gray-400 flex-shrink-0", className)}>
            <SideControl className="w-16 h-full flex-shrink-0" onSideItemClick={(label: string) => {
                setActivePanel(label);
            }} initActiveItem="Info" />
            {/* Render the active panel based on the selected item */}
            {activePanel === "Info" &&  <TrainInfoDisplay className="flex-1 flex-col w-full h-full" />}
            {activePanel === "Optimizer" && <OptimizerControl className="flex-1 flex-col w-full h-full" />}
            {activePanel === "Checkpoint" && <CheckpointControl className="flex-1 flex-col w-full h-full" />}
            {activePanel === "Dataset" && <div  className="flex-1 w-full h-full">Dataset Panel Content</div>}
        </div>
    );
};

export default ControlBar;