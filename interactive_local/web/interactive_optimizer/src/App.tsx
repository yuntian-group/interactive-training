import NavigationBar from "./components/navbar/navbar";
import ControlBar from "./components/controlBar/controlBar";
import MetricsPanel from "./components/metricsDisplay/metricsPanel";
import { useAppDispatch, useAppSelector } from "./hooks/userTypedHooks";
import { getOptimizerStateFromServer } from "./features/optimizerState/action";
import { getCheckpointStateFromServer } from "./features/checkpointState/action";
import { getTrainLogDataFromServer } from "./features/trainLogData/action";
import { getTrainInfoForInitializaiton } from "./features/trainInfo/actions";
import {getModelInfoFromServer} from "./features/modelInfo/action";
import { connectWebSocket, disconnectWebSocket } from "./features/trainEventWebsocket/actions";

import "./App.css";
import { useEffect } from "react";

function App() {
  const dispatch = useAppDispatch();

  const trainInfoStatus = useAppSelector(
    (state) =>  state.trainInfo.trainInfo.status
  );

  useEffect(() => {
    dispatch(getTrainInfoForInitializaiton());
  }, [dispatch]);

  useEffect(() => {
    // Trigger only once when the training starts
    if (trainInfoStatus === "running") {
      dispatch(getModelInfoFromServer());
      dispatch(getOptimizerStateFromServer());
      dispatch(getCheckpointStateFromServer());
      dispatch(getTrainLogDataFromServer());
      dispatch(connectWebSocket("ws://localhost:9876/ws/message/"));
      return () => {
        // Cleanup: disconnect the WebSocket when the component unmounts or training stops
        dispatch(disconnectWebSocket());
      }
    }
  }, [dispatch, trainInfoStatus]);

  return (
    <div className="App h-screen w-screen bg-gray-100 flex flex-col">
      <header className="App-header flex-shrink-0 h-12">
        <NavigationBar className="h-full" />
      </header>
      <div className="flex flex-row flex-1 overflow-auto w-full flex-shrink-0">
        <ControlBar className="h-full w-[24rem]" />
        <MetricsPanel className="h-full w-full overflow-auto" />
      </div>
    </div>
  );
}

export default App;
