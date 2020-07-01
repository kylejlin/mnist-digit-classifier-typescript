import { Option, Result } from "rusty-ts";
import { AccuracyRate } from "./data";
import { MnistData } from "./data/mnist";
import { Matrix } from "./matrix";
import { Network, WeightInitializationMethod } from "./network";
import { NetworkTester, NetworkTrainer } from "./networkServices";

export type AppState =
  | CreateNetworkState
  | NetworkMainMenuState
  | HyperParameterMenuState
  | TrainingInProgressState
  | TestState
  | ViewState
  | CropState
  | MnistLoadingErrorState;

export enum StateType {
  CreateNetwork,
  NetworkMainMenu,
  HyperParameterMenu,
  TrainingInProgress,
  Test,
  View,
  Crop,

  MnistLoadingError,
}

export interface StateMap {
  [StateType.CreateNetwork]: CreateNetworkState;
  [StateType.NetworkMainMenu]: NetworkMainMenuState;
  [StateType.HyperParameterMenu]: HyperParameterMenuState;
  [StateType.TrainingInProgress]: TrainingInProgressState;
  [StateType.Test]: TestState;
  [StateType.View]: ViewState;
  [StateType.Crop]: CropState;

  [StateType.MnistLoadingError]: MnistLoadingErrorState;
}

export interface CreateNetworkState {
  mnist: Option<MnistData>;

  stateType: StateType.CreateNetwork;

  hiddenLayerSizeInputValues: string[];
  weightInitializationMethod: WeightInitializationMethod;
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
  regularizationRateInputValue: string;
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

  uploadedImage: HTMLImageElement;
  darknessThreshold: number;
  cropSquare: Square;
  pendingCropAdjustment: Option<SquareAdjustment>;
  hoveredOverDraggable: Option<Draggable>;
  shouldInvertImage: boolean;
  labelInputValue: string;
}

export interface Square {
  x: number;
  y: number;
  size: number;
}

export interface SquareAdjustment {
  dragged: Draggable;
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
}

export enum Draggable {
  TopLeftCorner,
  TopRightCorner,
  BottomRightCorner,
  BottomLeftCorner,

  EntireSquare,
}

export type Corner =
  | Draggable.TopLeftCorner
  | Draggable.TopRightCorner
  | Draggable.BottomRightCorner
  | Draggable.BottomLeftCorner;

export interface MnistLoadingErrorState {
  mnist: Option<MnistData>;

  stateType: StateType.MnistLoadingError;

  errorMessage: string;
}
