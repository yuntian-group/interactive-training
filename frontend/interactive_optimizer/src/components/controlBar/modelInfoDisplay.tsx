import clsx from "clsx";
import { SimpleTreeView } from "@mui/x-tree-view/SimpleTreeView";
import { TreeItem } from "@mui/x-tree-view/TreeItem";
import { useAppSelector, useAppDispatch } from "../../hooks/userTypedHooks";
import type { ModelDataNode } from "../../features/modelInfo/type";
import { useMemo, useState } from "react";
import TextField from "@mui/material/TextField";

// Helper function to filter the tree recursively
const filterTree = (
  node: ModelDataNode,
  filterText: string
): ModelDataNode | null => {
  const lowerCaseFilter = filterText.toLowerCase();

  // Recursively filter children first
  const filteredChildren = node.children
    ?.map((child) => filterTree(child, filterText))
    .filter((child): child is ModelDataNode => child !== null);

  // A node is included if its name matches the filter OR if it has any children left after filtering
  if (
    node.name.toLowerCase().includes(lowerCaseFilter) ||
    (filteredChildren && filteredChildren.length > 0)
  ) {
    return { ...node, children: filteredChildren };
  }

  return null;
};

const displayTree = (node: ModelDataNode, dispatch: any): React.ReactNode => (
  <TreeItem
    key={node.name}
    itemId={node.name}
    label={
      <div className="flex">
        <span>{node.name}</span>
      </div>
    }
    onClick={(e) => {
      console.log("Selected layer:", node.name);
      dispatch({
        type: "modelInfoState/selectLayer",
        payload: node.name,
      });
      dispatch({
        type: "bottomDisplayState/addAndSetActiveTab",
        payload: "MODEL",
      });

      e.stopPropagation();
      e.preventDefault();
    }}
  >
    {node.children?.map((child) => displayTree(child, dispatch))}
  </TreeItem>
);

const TrainInfoDisplay: React.FC<{ className?: string }> = ({ className }) => {
  const appDispatch = useAppDispatch();
  const modelInfo = useAppSelector((state) => state.modelInfo.moduleTree);
  const [filterText, setFilterText] = useState("");

  const filteredModelInfo = useMemo(() => {
    if (!filterText.trim()) {
      return modelInfo;
    }
    return filterTree(modelInfo, filterText);
  }, [modelInfo, filterText]);

  const loadStatus = useAppSelector((state) => state.modelInfo.status);
  const loadDisplayInfo =
    loadStatus === "loading" ? (
      <p className="p-4 text-sm text-gray-500">Loading model info...</p>
    ) : loadStatus === "failed" ? (
      <p className="p-4 text-sm text-red-500">
        Failed to load model info. Please try again later.
      </p>
    ) : loadStatus === "idle" ? (
      <p className="p-4 text-sm text-gray-500">
        Model info not loaded yet. Please initiate a load.
      </p>
    ) : null;

  if (loadStatus !== "succeeded") {
    return (
      <div
        className={clsx("train-info-display flex flex-col h-full", className)}
      >
        <h4 className="flex-shrink-0 flex items-center justify-start text-left text-lg font-semibold p-2 bg-gray-200">
          Model Info
        </h4>
        {loadDisplayInfo}
      </div>
    );
  } else {
    return (
      <div
        className={clsx("train-info-display flex flex-col h-full", className)}
      >
        <h4 className="flex-shrink-0 flex items-center justify-start text-left text-lg font-semibold p-2 bg-gray-200">
          Model Info
        </h4>

        <TextField
          type="text"
          id="filter-layers"
          placeholder="Filter layers..."
          value={filterText}
          onChange={(e) => setFilterText(e.target.value)}
          variant="outlined"
          size="small"
          sx={{
            width: "100%",
            padding: "0.5rem",
            // remove up/down arrows if the browser still shows them
            "& input::-webkit-outer-spin-button, & input::-webkit-inner-spin-button":
              {
                WebkitAppearance: "none",
                margin: 0,
              },
            "& input[type=number]": {
              MozAppearance: "textfield",
            },
            "& .MuiOutlinedInput-root": {
              bgcolor: "#fff",
              borderRadius: 0,
              fontSize: "14px",
              justifyContent: "left",
              "& fieldset": { borderColor: "#000" },
              "&:hover fieldset": { borderColor: "#000" },
              "&.Mui-focused fieldset": { borderColor: "#000" },
            },
          }}
        />
        <div className="flex-grow overflow-auto">
          {filteredModelInfo ? (
            <SimpleTreeView
              sx={{
                // make long labels wrap/break
                "& .MuiTreeItem-content .MuiTreeItem-label": {
                  whiteSpace: "normal",
                  wordBreak: "break-word",
                  fontSize: "0.8rem",
                },
                "& .MuiTreeItem-content.Mui-focused, .MuiTreeItem-content.Mui-selected, & .MuiTreeItem-content.Mui-focusVisible":
                  {
                    borderRadius: 0,
                  },
              }}
            >
              {displayTree(filteredModelInfo, appDispatch)}
            </SimpleTreeView>
          ) : (
            <p className="p-4 text-sm text-gray-500">
              No matching layers found.
            </p>
          )}
        </div>
      </div>
    );
  }
};
export default TrainInfoDisplay;
