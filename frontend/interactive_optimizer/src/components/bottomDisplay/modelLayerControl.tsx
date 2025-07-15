import React from "react";
import clsx from "clsx";
import { useAppSelector } from "../../hooks/userTypedHooks";
import type {
  ModelDataNode,
  ModelHyperparameter,
  ModelOperation,
} from "../../features/modelInfo/type";
import {
  InfoField,
  NumericalInputControlWithSlider,
  RemoteFunctionControl,
} from "../sharedControl/sharedControl";
import { postTrainCommand } from "../../features/trainCommand/actions";
import { useAppDispatch } from "../../hooks/userTypedHooks";
import type { TrainCommandData } from "../../features/trainCommand/types";
import { v4 as uuidv4 } from "uuid";

export interface InfoDisplayProps {
  fields: Array<{ label: string; value: string }>;
}

export const InfoDisplay: React.FC<InfoDisplayProps> = ({ fields }) => (
  <div className="space-y-2">
    <div className="text-md font-medium text-gray-700">Layer Information</div>
    <div className="grid grid-cols-4 xs:grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-1">
      {fields.map((field, index) => (
        <InfoField
          key={index}
          label={field.label}
          value={field.value}
          className=" bg-white border border-gray-200"
        />
      ))}
    </div>
  </div>
);

export interface ControlSectionProps {
  layerName: string;
  operators: ModelOperation[];
  hyperparameters: ModelHyperparameter[];
  className?: string;
}

const ControlSection: React.FC<ControlSectionProps> = ({
  layerName,
  operators = [],
  hyperparameters = [],
  className = "",
}) => {

    const dispatch = useAppDispatch();

  const buildAndPostModelLayerParamUpdateCommand = (
    layerName: string,
    paramName: string,
    value: any
  ) => {
    // Post
    dispatch(postTrainCommand({
      command: "model_layer_parameter_update",
      args: JSON.stringify({
        layer_name: layerName,
        param_name: paramName,
        value: value,
      }),
      uuid: uuidv4(),
      time: Date.now(),
      status: "requested",
    } as TrainCommandData));
  };

  const buildAndPostModelLayerOperationCommand = (
    layerName: string,
    operationName: string,
    params: Record<string, any>
  ) => {
    dispatch(postTrainCommand({
      command: "model_layer_operation",
      args: JSON.stringify({
        layer_name: layerName,
        operation_name: operationName,
        params: params,
      }),
      uuid: uuidv4(),
      time: Date.now(),
      status: "requested",
    } as TrainCommandData));
  };

  return (
    <div className={clsx("control-section space-y-4", className)}>
      {operators.length > 0 && (
        <h3 className="text-md font-medium text-gray-700">Operations</h3>
      )}

      <div className="operators-section grid grid-cols-4 xs:grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-1">
        {operators.length > 0 &&
          operators.map((fInfo) => (
            <RemoteFunctionControl
              functionInfo={fInfo}
              key={fInfo.name}
              onApply={(params) =>
                buildAndPostModelLayerOperationCommand(layerName, fInfo.name, params)
              }
              className="text-black p-2 bg-white"
            />
          ))}
      </div>
      {hyperparameters.length > 0 && (
        <h3 className="text-md font-medium text-gray-700">Parameters</h3>
      )}

      {hyperparameters.length > 0 && (
        <div className="flex gap-1 grid grid-cols-4 xs:grid-cols-1 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
          {hyperparameters
            .filter(
              (hparam) => hparam.type === "int" || hparam.type === "float"
            ) // temp make it numerical only for now
            .map((hparam) => (
              <NumericalInputControlWithSlider
                key={hparam.name}
                id={hparam.name}
                label={hparam.name}
                value={hparam.value}
                min={0} // Assuming a default min, adjust as needed
                max={1.0} // Assuming a default max, adjust as needed
                step={0.01} // Assuming a default step, adjust as needed
                onChange={() => {}}
                onApply={(value) => {
                  buildAndPostModelLayerParamUpdateCommand(
                    layerName,
                    hparam.name,
                    value
                  );
                }}
                className="text-black bg-white border border-gray-200"
              />
            ))}
        </div>
      )}
    </div>
  );
};

interface ModelLayerControlProps {
  className: string;
  infoFields?: Array<{ label: string; value: string }>;
}

const ModelLayerControl: React.FC<ModelLayerControlProps> = ({ className }) => {
  const selectedLayer = useAppSelector(
    (state) => state.modelInfo.selectedLayer
  );
  const layerInfo = useAppSelector(
    (state) =>
      state.modelInfo.nodeMap[selectedLayer] ||
      ({
        name: selectedLayer,
        moduleType: "Unknown",
        children: [],
        operators: [],
        hyperparameters: [],
      } as ModelDataNode)
  );

  const infoFields = [
    { label: "Layer Name", value: layerInfo.name || "Unknown" },
    { label: "Layer Type", value: layerInfo.moduleType || "Unknown" },
  ];

  return (
    <div
      className={clsx(
        "modellayer p-3 bg-white border border-gray-200 shadow-sm flex flex-col space-y-4 flex-shrink-0 min-w-0",
        className
      )}
    >
      <InfoDisplay fields={infoFields} />
      <ControlSection
        layerName={layerInfo.name}
        operators={layerInfo.operators || []}
        hyperparameters={layerInfo.hyperparameters || []}
      />
    </div>
  );
};

export default ModelLayerControl;
