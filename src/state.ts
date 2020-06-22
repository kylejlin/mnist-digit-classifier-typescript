import { Option, Result } from "rusty-ts";
import { AccuracyRate } from "./data";
import { MnistData } from "./data/mnist";
import { Network } from "./network";
import { NetworkTester, NetworkTrainer } from "./networkServices";
import { Matrix } from "./matrix";

export type AppState =
  | CreateNetworkState
  | NetworkMainMenuState
  | HyperParameterMenuState
  | TrainingInProgressState
  | TestState
  | ViewState
  | CropState;

export enum StateType {
  CreateNetwork,
  NetworkMainMenu,
  HyperParameterMenu,
  TrainingInProgress,
  Test,
  View,
  Crop,
}

export interface StateMap {
  [StateType.CreateNetwork]: CreateNetworkState;
  [StateType.NetworkMainMenu]: NetworkMainMenuState;
  [StateType.HyperParameterMenu]: HyperParameterMenuState;
  [StateType.TrainingInProgress]: TrainingInProgressState;
  [StateType.Test]: TestState;
  [StateType.View]: ViewState;
  [StateType.Crop]: CropState;
}

export interface CreateNetworkState {
  mnist: Option<MnistData>;

  stateType: StateType.CreateNetwork;

  hiddenLayerSizeInputValues: string[];
  previousNetwork: Option<Network>;
}

export interface NetworkMainMenuState {
  mnist: Option<MnistData>;

  stateType: StateType.NetworkMainMenu;

  network: Network;
}

export interface HyperParameterMenuState {
  mnist: Option<MnistData>;

  stateType: StateType.HyperParameterMenu;

  network: Network;

  batchSizeInputValue: string;
  epochsInputValue: string;
  learningRateInputValue: string;
}

export interface TrainingInProgressState {
  mnist: Option<MnistData>;

  stateType: StateType.TrainingInProgress;

  network: Network;

  networkTrainer: NetworkTrainer;
  epochAccuracyRates: EpochAccuracyRate[];
}

export interface EpochAccuracyRate extends AccuracyRate {
  epoch: number;
}

export interface TestState {
  mnist: Option<MnistData>;

  stateType: StateType.Test;

  network: Network;

  accuracyRate: Result<AccuracyRate, NetworkTester>;
}

export interface ViewState {
  mnist: Option<MnistData>;

  stateType: StateType.View;

  network: Network;

  viewedIndex: number;
  customImages: CustomImage[];
}

export interface CustomImage {
  u8Matrix: Matrix;
  label: number;
}

export interface CropState {
  mnist: Option<MnistData>;

  stateType: StateType.Crop;

  network: Network;

  customImages: CustomImage[];
}
