// src/components/metricsDisplay/metricsDisplayChart.tsx
import clsx from "clsx";
import React from "react";
import ReactECharts from "echarts-for-react";

interface MetricsDisplayChartProps {
  className?: string;
  title: string;
  branchNames: string[];
  branchDataList: number[][][]; // correspond to [branchName, globalStep, value]
}

const MetricsDisplayChart: React.FC<MetricsDisplayChartProps> = ({
  className,
  title,
  branchNames,
  branchDataList,
}) => {
  const legendHeight =
    branchNames.length > 0 ? Math.ceil(branchNames.length / 4) * 20 + 10 : 30;

  const options = {
    grid: {
      top: 40,
      bottom: legendHeight + 2, // Adaptive bottom margin based on legend
      left: 0,
      right: 0,
      containLabel: true,
    },
    title: {
      text: title || "metrics",
      left: "center",
      textStyle: {
        fontSize: 12,
        fontWeight: "bold",
      },
    },
    tooltip: {
      trigger: "axis",
    },
    xAxis: {
      type: "value",
      name: "Step",
      axisLabel: {
        interval: "auto", // Automatically determine interval
        rotate: 0, // Keep horizontal, but will rotate if needed
        formatter: (value: number) => {
          // Format large numbers for better readability
          if (value !== 0 && Math.abs(value) < 0.001) {
            return value.toExponential(2);
          }
          if (value >= 1000000) {
            return (value / 1000000).toFixed(1) + "M";
          }
          if (value >= 1000) {
            return (value / 1000).toFixed(1) + "K";
          }
          return value.toString();
        },
      },
      splitNumber: 5, // Limit number of ticks
    },
    yAxis: {
      type: "value",
      name: "Value",
      axisLabel: {
        interval: "auto",
        formatter: (value: number) => {
          if (value !== 0 && Math.abs(value) < 0.001) {
            return value.toExponential(2);
          }
          if (Math.abs(value) >= 1000000) {
            return (value / 1000000).toFixed(1) + "M";
          }
          if (Math.abs(value) >= 1000) {
            return (value / 1000).toFixed(1) + "K";
          }
          return value.toFixed(2);
        },
      },
      splitNumber: 5,
    },
    series: branchNames.map((branchName, index) => ({
      name: branchName || `Branch ${index + 1}`,
      type: "line",
      data: branchDataList[index] || [],
      showSymbol: false,
      hoverAnimation: false,
      smooth: false,
      lineStyle: {
        width: 1.5,
      },
    })),
    legend: {
      data: branchNames.map(
        (branchName, index) => branchName || `Branch ${index + 1}`
      ),
      orient: "horizontal",
      left: "center",
      bottom: 5, // Closer to x-axis
      itemWidth: 10,
      itemHeight: 10,
      textStyle: {
        fontSize: 10,
      },
      type: "scroll",
      pageButtonItemGap: 5,
      pageButtonGap: 10,
    },
    animation: false,
  };

  return (
    <div className={clsx("metrics-display-chart", className)}>
      <ReactECharts
        option={options}
        style={{ height: "100%", width: "100%" }}
      />
    </div>
  );
};

export default MetricsDisplayChart;
