import { createSlice } from "@reduxjs/toolkit";

import type bottomDisplayState from "./type";

const initialState: bottomDisplayState = {
  tabs: ["LOG"],
  activeTab: "LOG", // Default active tab
  height: "250px", // Initial height of the bottom display
};

const bottomDisplayStateSlice = createSlice({
  name: "bottomDisplayState",
  initialState,
  reducers: {
    addTab: (state, action) => {
      if (state.tabs.includes(action.payload)) {
        return;
      }
      state.tabs.push(action.payload);
    },
    addAndSetActiveTab: (state, action) => {
      if (state.tabs.includes(action.payload)) {
        state.activeTab = action.payload;
        return;
      } else {
        state.tabs.push(action.payload);
        state.activeTab = action.payload;
      }
    },
    setActiveTab: (state, action) => {
      state.activeTab = action.payload;
    },
    removeTab: (state, action) => {
      state.tabs = state.tabs.filter((tab) => tab !== action.payload);
    },
    clearTabs: (state) => {
      state.tabs = ["LOG"];
      state.activeTab = "LOG"; // Reset to default active tab
    },
    setHeight: (state, action) => {
      state.height = action.payload;
    },
  },
});

export const {
  addTab,
  addAndSetActiveTab,
  setActiveTab,
  removeTab,
  clearTabs,
  setHeight,
} = bottomDisplayStateSlice.actions;
export default bottomDisplayStateSlice.reducer;
