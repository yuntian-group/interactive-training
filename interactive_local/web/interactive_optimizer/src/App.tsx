import NavigationBar from "./components/navbar/navbar";
import ControlBar from "./components/controlBar/controlBar";
import { useAppDispatch, useAppSelector } from "./hooks/userTypedHooks";
import { getOptimizerStateFromServer } from "./features/optimizerState/action";
import { getCheckpointStateFromServer } from "./features/checkpointState/action";
import { getTrainInfoForInitializaiton } from "./features/trainInfo/actions";
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
      dispatch(getOptimizerStateFromServer());
      dispatch(getCheckpointStateFromServer());
    }
  }, [dispatch, trainInfoStatus]);

  return (
    <div className="App h-screen w-screen bg-gray-100 flex flex-col">
      <header className="App-header flex-shrink-0 h-12">
        <NavigationBar className="h-full" />
      </header>
      <div className="flex flex-row flex-1 overflow-auto w-90">
        <ControlBar className="h-full w-full" />
      </div>
    </div>
  );
}

export default App;
