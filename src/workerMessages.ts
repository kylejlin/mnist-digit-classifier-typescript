import { AccuracyRate } from "./data";
import { StochasticGradientDescentHyperParameters } from "./network";

export enum WorkerMessageType {
  StartTrainingRequest,
  TrainingEpochCompleteNotification,
  TerminateTrainingRequest,
  TerminateTrainingResponse,

  StartTestingRequest,
  TestCompleteNotification,
}

export type NetworkTrainerRequest =
  | StartTrainingRequest
  | TerminateTrainingRequest;

export type NetworkTrainerNotification =
  | TrainingEpochCompleteNotification
  | TerminateTrainingResponse;

export interface StartTrainingRequest {
  messageType: WorkerMessageType.StartTrainingRequest;

  networkBuffer: ArrayBuffer;
  hyperParams: StochasticGradientDescentHyperParameters;
}

export interface TrainingEpochCompleteNotification {
  messageType: WorkerMessageType.TrainingEpochCompleteNotification;

  accuracyRate: AccuracyRate;
  epoch: number;
}

export interface TerminateTrainingRequest {
  messageType: WorkerMessageType.TerminateTrainingRequest;
}

export interface TerminateTrainingResponse {
  messageType: WorkerMessageType.TerminateTrainingResponse;

  networkBuffer: ArrayBuffer;
}

export interface StartTestingRequest {
  messageType: WorkerMessageType.StartTestingRequest;

  networkBuffer: ArrayBuffer;
}

export interface TestCompleteNotification {
  messageType: WorkerMessageType.TestCompleteNotification;
  accuracyRate: AccuracyRate;
}
