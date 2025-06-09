import clsx from "clsx";
import React, {useState } from "react";
import SideControl from "./sideControl";
import OptimizerControl from "./optimizerControl";

type Props = React.HTMLAttributes<HTMLDivElement>;

const ControlBar: React.FC<Props> = ({ className }: Props) => {

    const [activePanel, setActivePanel] = useState<string>("Info");

    return (
        <div className={clsx("control-bar bg-white flex flex-row", className)}>
            <SideControl className="w-16 h-full" onSideItemClick={(label: string) => {
                setActivePanel(label);
            }} initActiveItem="Info" />
            {/* Render the active panel based on the selected item */}
            {activePanel === "Info" && <div className="flex-1 w-full h-full">Info Panel Content</div>}
            {activePanel === "Optimizer" && <OptimizerControl className="w-full h-full" />}
            {activePanel === "Checkpoint" && <div  className="flex-1 w-full h-full">Checkpoint Panel Content</div>}
            {activePanel === "Dataset" && <div  className="flex-1 w-full h-full">Dataset Panel Content</div>}
        </div>
    );
};

export default ControlBar;