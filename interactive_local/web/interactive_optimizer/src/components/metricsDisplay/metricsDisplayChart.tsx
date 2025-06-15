// src/components/metricsDisplay/metricsDisplayChart.tsx
import clsx from "clsx";
import React from "react";
import ReactECharts from "echarts-for-react";

interface MetricsDisplayChartProps {
    className?: string;
    title?: string;
    xAxisData?: string[] | number[];
    yAxisData?: number[];
}

const MetricsDisplayChart: React.FC<MetricsDisplayChartProps> = ({ className, title, xAxisData, yAxisData }) => {
    const options = {
        grid: {
          top: 40,      // shrink top margin
          bottom: 10,   // shrink bottom margin
          left: 0,     // optional: move y-axis in a bit
          right: 0,    // optional: move series away from right edge
          containLabel: true  // make sure labels still fit
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
            type: "category",
            data: xAxisData || [],
        },
        yAxis: {
            type: "value",
        },
        series: [
            {
                text: title || "metrics",
                type: "line",
                data: yAxisData || [],
                showSymbol: false, 
                lineStyle: {
                    width: 1  // default is usually 2, so 1 makes it thinner
                }
            }
        ],
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