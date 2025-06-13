import { createSlice } from "@reduxjs/toolkit";
import type { trainEventState } from "./types";

const initialState: trainEventState = {
  connected: false,
};

const trainEventSlice = createSlice({
  name: "trainEvent",
  initialState,
  reducers: {},
  extraReducers: (builder) => {
    builder
      // .addMatcher(
      //   (action): action is PayloadAction<any> => action.type === "WS_RECEIVED",
      //   (state, action) => {
      //     console.log(
      //       "Received WebSocket message:",
      //       action.payload,
      //       state.messages
      //     );
      //     state.messages.push(action.payload);
      //   }
      // )
      .addMatcher(
        (action) => action.type === "WS_CONNECT",
        (state) => {
          state.connected = true;
        }
      )
      .addMatcher(
        (action) => action.type === "WS_CLOSED",
        (state) => {
          state.connected = false;
        }
      );
  },
});

export default trainEventSlice.reducer;
