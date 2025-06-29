import { createSlice } from "@reduxjs/toolkit";

import type bottomDisplayState from "./type";

const initialState: bottomDisplayState = {
  tabs: ["LOG", "MODEL"],
  activeTab: "MODEL", // Default active tab
  height: "250px", // Initial height of the bottom display
};

const bottomDisplayStateSlice = createSlice({
  name: "bottomDisplayState",
  initialState,
  reducers: {
    addTabReducer: (state, action) => {
      if (state.tabs.includes(action.payload)) {
        return;
      }
      state.tabs.push(action.payload);
    },
    addAndSetActiveTabReducer: (state, action) => {
      if (state.tabs.includes(action.payload)) {
        state.activeTab = action.payload;
        return;
      } else {
        state.tabs.push(action.payload);
        state.activeTab = action.payload;
      }
    },
    setActiveTabReducer: (state, action) => {
      state.activeTab = action.payload;
    },
    removeTabReducer: (state, action) => {
      state.tabs = state.tabs.filter((tab) => tab !== action.payload);
    },
    clearTabsReducer: (state) => {
      state.tabs = ["LOG"];
      state.activeTab = "LOG"; // Reset to default active tab
    },
    setHeightReducer: (state, action) => {
      state.height = action.payload;
    },
  },
});

export const { addTabReducer } = bottomDisplayStateSlice.actions;
export default bottomDisplayStateSlice.reducer;
