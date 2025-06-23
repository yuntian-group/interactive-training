import { configureStore } from "@reduxjs/toolkit";
import trainEventReducer from "../features/trainEventWebsocket/reducer";
import { websocketMiddleware } from "../features/trainEventWebsocket/middleware";
import trainCommandReducer from "../features/trainCommand/reducer";
import trainInfoReducer from "../features/trainInfo/reducer";
import optimizerStateReducer from "../features/optimizerState/reducer";
import checkpointStateReducer from "../features/checkpointState/reducer";
import TrainLogDataReducer from "../features/trainLogData/reducer";
import ModelInfoReducer from "../features/modelInfo/reducer";
// Define the store
export const store = configureStore({
  reducer: {
    trainEvent: trainEventReducer,
    trainCommand: trainCommandReducer,
    trainInfo: trainInfoReducer,
    modelInfo: ModelInfoReducer,
    optimizerState: optimizerStateReducer,
    checkpointState: checkpointStateReducer,
    trainLogData: TrainLogDataReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware().concat(websocketMiddleware),
});

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
