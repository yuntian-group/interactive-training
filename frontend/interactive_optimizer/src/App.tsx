import NavigationBar from "./components/navbar/navbar";
import ControlBar from "./components/controlBar/controlBar";

import MetricsPanel from "./components/metricsDisplay/metricsPanel";
import BottomDisplay from "./components/bottomDisplay/bottomDisplay";
import { getModelInfoFromServer } from "./features/modelInfo/action";
import { useAppDispatch, useAppSelector } from "./hooks/userTypedHooks";
import { getTrainLogDataFromServer } from "./features/trainLogData/action";
import { getTrainInfoForInitializaiton } from "./features/trainInfo/actions";
import { getOptimizerStateFromServer } from "./features/optimizerState/action";
import { getCheckpointStateFromServer } from "./features/checkpointState/action";
import { getDatasetInfoFromServer } from "./features/datasetInfo/action";


import {
  connectWebSocket,
  disconnectWebSocket,
} from "./features/trainEventWebsocket/actions";

import "./App.css";
import { useEffect, useRef, useState } from "react";
import { Resizable } from "re-resizable";
import { toPixels } from "./utils/cssLength";
import {websocketHost} from "./api/api";


function App() {
  const dispatch = useAppDispatch();
  const trainInfoStatus = useAppSelector(
    (state) => state.trainInfo.trainInfo.status
  );

  const thresHeight = 100;
  const minHeight = 0;
  const handlerWidth = 4;
  const mainDisplayRef = useRef<HTMLDivElement>(null);
  const bottomHeight = useAppSelector(
    (state) => state.bottomDisplayState.height
  );

  const [size, setSize] = useState<{ width: string; height: number | string }>({
    width: "100%",
    height: bottomHeight, // Initial height set to threshold height
  });

  const handleBottomResize = (
    _e: MouseEvent | TouchEvent, // The resize event
    _direction:
      | "top"
      | "right"
      | "bottom"
      | "left"
      | "topRight"
      | "bottomRight"
      | "bottomLeft"
      | "topLeft", // The direction of resize
    ref: HTMLElement // The resized element (DOM node)
  ) => {
    if (!mainDisplayRef.current) {
      return;
    }
    let curHeight = toPixels(ref.style.height, mainDisplayRef.current);
    const oldHeight = toPixels(bottomHeight, mainDisplayRef.current);

    if (curHeight <= thresHeight) {
      if (curHeight <= oldHeight) {
        curHeight = 0;
        localStorage.setItem("bottomDisplayHeight", oldHeight.toString());
      } else {
        curHeight = thresHeight;
        localStorage.setItem("bottomDisplayHeight", thresHeight.toString());
      }
    }
    dispatch({
      type: "bottomDisplayState/setHeightReducer",
      payload: curHeight,
    });
  };

  useEffect(() => {
    dispatch(getTrainInfoForInitializaiton());
  }, [dispatch]);

  useEffect(() => {
    // If the bottom display is active, set the height to 50% of the viewport height
    console.log(
      "Setting bottom display height based on local storage or threshold height."
    );
    if (!mainDisplayRef.current) {
      return;
    }
    if (bottomHeight !== 0) {
      localStorage.setItem("bottomDisplayHeight", bottomHeight.toString());
    }
    setSize({
      width: "100%",
      height: toPixels(bottomHeight, mainDisplayRef.current!),
    });
  }, [bottomHeight, dispatch]);

  const hasConnectedRef = useRef(false);

  useEffect(() => {
    // Trigger only once when the training starts for the first time
    if (trainInfoStatus === "running" && !hasConnectedRef.current) {
      console.log(
        "Training started, initializing data fetch and WebSocket connection."
      );
      dispatch(getModelInfoFromServer());
      dispatch(getOptimizerStateFromServer());
      dispatch(getCheckpointStateFromServer());
      dispatch(getTrainLogDataFromServer());
      dispatch(getDatasetInfoFromServer());
      dispatch(connectWebSocket(websocketHost));
      hasConnectedRef.current = true;
    }
  }, [dispatch, trainInfoStatus]);

  useEffect(() => {
    // Cleanup: disconnect the WebSocket only when the component unmounts (page refresh/navigation)
    return () => {
      if (hasConnectedRef.current) {
        dispatch(disconnectWebSocket());
      }
    };
  }, [dispatch]);

  return (
    <div className="App h-screen w-screen bg-gray-100 flex flex-col">
      <header className="App-header flex-shrink-0 h-12">
        <NavigationBar className="h-full" />
      </header>
      <div
        ref={mainDisplayRef}
        className="flex flex-row flex-1 overflow-auto w-full flex-shrink-0"
      >
        <ControlBar className="h-full w-[24rem]" />
        <div className="mainDisplay flex flex-col flex-1 overflow-auto w-full flex-shrink-0 min-h-0 box-border">
          <MetricsPanel className="flex-1 w-full overflow-auto box-border" />
          <Resizable
            defaultSize={{ width: "100%", height: thresHeight }}
            size={size}
            onResizeStop={handleBottomResize}
            onResize={(_e: MouseEvent | TouchEvent, _direction, ref) => {
              setSize((prevSize) => ({
                ...prevSize,
                height: toPixels(ref.style.height, mainDisplayRef.current!),
              }));
            }}
            minHeight={minHeight}
            maxHeight="50%"
            enable={{
              top: true,
              right: false,
              bottom: false,
              left: false,
              topRight: false,
              bottomRight: false,
              bottomLeft: false,
              topLeft: false,
            }}
            handleClasses={{
              top: "cursor-row-resize border-t-[4px] border-gray-300 hover:border-blue-500",
            }}
            handleStyles={{
              top: {
                top: -handlerWidth, // Adjust the top position to align with the top border
                height: 0, // Set height to 0 to avoid extra space
              },
            }}
          >
            {toPixels(size.height) > minHeight && (
              <BottomDisplay className="h-full w-full overflow-auto" />
            )}
          </Resizable>
        </div>
      </div>
    </div>
  );
}

export default App;
