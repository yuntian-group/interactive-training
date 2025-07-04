import type { SingleMetricsPoint } from "./type";
import type { TrainCommandData } from "../trainCommand/types";

export interface SingleMetricsBuffer {
  localSteps: Float64Array;
  globalSteps: Float64Array;
  wallTimes: Float64Array;
  vals: Float32Array;
  len: number;
}

const defaultInitialArraySize = 1000;

const metricsBranchBuffer: Map<
  string,
  Map<string, SingleMetricsBuffer>
> = new Map<string, Map<string, SingleMetricsBuffer>>();

function ensureMetrics(metricName: string): Map<string, SingleMetricsBuffer> {
  if (!metricsBranchBuffer.has(metricName)) {
    metricsBranchBuffer.set(metricName, new Map<string, SingleMetricsBuffer>());
  }
  return metricsBranchBuffer.get(metricName)!;
}

function ensureMetricBuffer(
  branchId: string,
  metricName: string,
  initialArraySize: number = defaultInitialArraySize
): SingleMetricsBuffer {
  const branchesUnderMetric = ensureMetrics(metricName);

  if (!branchesUnderMetric.has(branchId)) {
    branchesUnderMetric.set(branchId, {
      localSteps: new Float64Array(initialArraySize),
      globalSteps: new Float64Array(initialArraySize),
      wallTimes: new Float64Array(initialArraySize),
      vals: new Float32Array(initialArraySize),
      len: 0,
    });
  }
  let ret = branchesUnderMetric.get(branchId)!;
  if (ret.len == ret.localSteps.length) {
    // Resize the arrays if they are full
    const newSize = ret.len * 2;
    ret = {
      localSteps: new Float64Array(newSize),
      globalSteps: new Float64Array(newSize),
      wallTimes: new Float64Array(newSize),
      vals: new Float32Array(newSize),
      len: ret.len,
    };

    const allMetricsUnderBranch = branchesUnderMetric.get(metricName)!;

    // Copy the old data to the new arrays
    ret.localSteps.set(allMetricsUnderBranch.localSteps);
    ret.globalSteps.set(allMetricsUnderBranch.globalSteps);
    ret.wallTimes.set(allMetricsUnderBranch.wallTimes);
    ret.vals.set(allMetricsUnderBranch.vals);
    branchesUnderMetric.set(metricName, ret);
  }
  return ret;
}

export function appendNewDataPoint(singleUpdate: SingleMetricsPoint) {
  const branchId = singleUpdate.branch_id;
  const metrics = singleUpdate.metrics;
  const wallTime = singleUpdate.wall_time;

  for (const [metricName, value] of Object.entries(metrics)) {
    const metricBuffer = ensureMetricBuffer(branchId, metricName);
    metricBuffer.localSteps[metricBuffer.len] = singleUpdate.local_step;

    if (!singleUpdate.metrics.global_step) {
      metricBuffer.globalSteps[metricBuffer.len] = singleUpdate.local_step;
    } else {
      metricBuffer.globalSteps[metricBuffer.len] =
        singleUpdate.metrics.global_step;
    }
    metricBuffer.wallTimes[metricBuffer.len] = wallTime;
    metricBuffer.vals[metricBuffer.len] = value;
    metricBuffer.len++;
  }
}

export function loadInitialSnapshot(
  metricsRecord: Record<string, SingleMetricsPoint[]>
) {
  if (Object.keys(metricsRecord).length === 0) {
    return;
  }

  for (const branchId of Object.keys(metricsRecord)) {
    const branchMetrics: SingleMetricsPoint[] = metricsRecord[branchId];

    if (!branchMetrics || branchMetrics.length === 0) {
      continue; // Skip empty branches
    }

    const initialSize = Math.max(
      defaultInitialArraySize,
      branchMetrics.length * 2 + 2
    );

    for (const singleUpdate of branchMetrics) {
      for (const [metricName, value] of Object.entries(singleUpdate.metrics)) {
        const metricBuffer = ensureMetricBuffer(
          branchId,
          metricName,
          initialSize
        );
        metricBuffer.localSteps[metricBuffer.len] = singleUpdate.local_step;
        metricBuffer.globalSteps[metricBuffer.len] =
          singleUpdate.metrics.global_step || singleUpdate.local_step;
        metricBuffer.wallTimes[metricBuffer.len] = singleUpdate.wall_time;
        metricBuffer.vals[metricBuffer.len] = value;
        metricBuffer.len++;
      }
    }
  }
}

export default metricsBranchBuffer;
