import clsx from "clsx";
import * as React from "react";
import { X } from "lucide-react";
import { useAppSelector, useAppDispatch } from "../../hooks/userTypedHooks";
import LogDisplay from "./logDisplay";
import ModelLayerControl from "./modelLayerControl";

interface TabsProps {
  onChange?: (tab: string, index: number) => void;
  className?: string;
}

export const Tabs: React.FC<TabsProps> = ({ onChange, className }) => {
  const dispatch = useAppDispatch();

  const tabs = useAppSelector((state) => state.bottomDisplayState.tabs);
  const activeTab = useAppSelector(
    (state) => state.bottomDisplayState.activeTab
  );

  const handleSelect = (tab: string, idx: number) => {
    dispatch({ type: "bottomDisplayState/setActiveTabReducer", payload: tab });
    onChange?.(tab, idx);
  };

  const onKeyDown = (e: React.KeyboardEvent, tab: string, idx: number) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      handleSelect(tab, idx);
    }
  };

  return (
    <div className={clsx(className, "justify-between flex flex-row-0 flex-shrink-0")}>
      <div role="tablist" className="flex items-center w-full flex-grow-0">
        {tabs.map((tab, idx) => (
          <div
            key={tab}
            role="tab"
            tabIndex={0}
            aria-selected={tab === activeTab}
            onClick={() => handleSelect(tab, idx)}
            onKeyDown={(e) => onKeyDown(e, tab, idx)}
            className={clsx(
              "px-4 transition cursor-pointer h-full text-xs  align-middle flex items-center justify-center",
              tab === activeTab
                ? "bg-white border-b-2"
                : "bg-gray-100 hover:bg-gray-200"
            )}
          >
            {tab}
          </div>
        ))}
      </div>
      <button
        className="px-2 bg-transparent text-black transition h-full align-middle flex items-center justify-center"
        onClick={() => {
          dispatch({ type: "bottomDisplayState/setHeightReducer", payload: 0 });
        }}
      >
        <X size={"16px"} />
      </button>
    </div>
  );
};

const BottomBar: React.FC<{ className?: string }> = ({ className }) => {
  const handleTabChange = (tab: string, idx: number) => {
    console.log("active tab:", tab, idx);
  };

  const activeTab = useAppSelector(
    (state) => state.bottomDisplayState.activeTab
  );

  return (
    <div className={clsx("bottom-display flex flex-col h-full", className)}>
      <Tabs onChange={handleTabChange} className="bg-gray-50 h-8" />
      <div className="flex-1 min-h-0">
        {activeTab === "MODEL" && (
          <ModelLayerControl className="h-full overflow-auto p-4" />
        )}
        {activeTab === "LOG" && (
          <LogDisplay className="h-full overflow-auto p-4" />
        )}
      </div>
    </div>
  );
};

export default BottomBar;
