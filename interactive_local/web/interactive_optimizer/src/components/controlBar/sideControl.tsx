import clsx from "clsx";
import React, { useState } from "react";
import { SlidersHorizontal, Database, Info, Save, Brain } from "lucide-react"; // Or replace with any icon set you use

type MenuItem = {
  label: string;
  icon: React.ReactNode;
  active?: boolean;
};

const menuItems: MenuItem[] = [
  { label: "Info", icon: <Info size={20} /> },
  { label: "Optimizer", icon: <SlidersHorizontal size={20} /> },
  { label: "Model", icon: <Brain size={20} /> },
  { label: "Checkpoint", icon: <Save size={20} /> },
  { label: "Dataset", icon: <Database size={20} /> },
];

type Props = React.HTMLAttributes<HTMLDivElement> & {
  initActiveItem: string; // Optional prop to set initial active item
  onSideItemClick: (label: string) => void;
};

const SideControl: React.FC<Props> = ({
  className,
  initActiveItem,
  onSideItemClick,
  ...rest
}: Props) => {
  const [activeItem, setActiveItem] = useState<string>(initActiveItem);

  return (
    <div
      className={clsx(
        "bg-white border-r flex flex-col items-center py-6 space-y-8 border-gray-200",
        className
      )}
    >
      {/* Logo or Title */}
      {menuItems.map(({ label, icon }) => (
        <div
          key={label}
          onClick={() => {
            // Handle click event, e.g., set active state or navigate
            setActiveItem(label);
            onSideItemClick(label); // Call the provided callback with the label
          }}
          className={`flex flex-col items-center w-full text-[10px] font-medium space-y-1
                ${
                  activeItem === label
                    ? "bg-sky-50 text-sky-600 px-3 py-2"
                    : "text-gray-700 px-3 py-2 hover:bg-gray-100 hover:text-sky-600"
                }`}
        >
          <div>{icon}</div>
          <div>{label}</div>
        </div>
      ))}
    </div>
  );
};

export default SideControl;
