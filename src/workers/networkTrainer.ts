import { AccuracyRate, LabeledImage, VectorLabeledImage } from "../data";
import { mnistProm } from "../data/mnist";
import { Network } from "../network";
import { deserializeNetwork, serializeNetwork } from "../networkSerializer";
import {
  NetworkTrainerNotification,
  NetworkTrainerRequest,
  StartTrainingRequest,
  StochasticGradientDescentHyperParameters,
  TerminateTrainingResponse,
  TrainingEpochCompleteNotification,
  WorkerMessageType,
} from "../workerMessages";
import { Option, option } from "rusty-ts";

interface Self {
  postMessage(
    message: NetworkTrainerNotification,
    transfers?: Transferable[]
  ): void;
  addEventListener: Worker["addEventListener"];
}

declare const self: Self;

export {};

let shouldTerminateAfterThisEpoch = false;
let trainedNetwork: Option<Network> = option.none();

self.addEventListener("message", (event) => {
  const { data } = event;

  if (data !== null && "object" === typeof data && "messageType" in data) {
    const message: NetworkTrainerRequest = data;
    switch (message.messageType) {
      case WorkerMessageType.StartTrainingRequest:
        startTrainingOnceMnistLoads(message);
        break;
      case WorkerMessageType.TerminateTrainingRequest:
        terminateTrainingAfterThisEpoch();
        break;

      default: {
        // Force exhaustive matching

        // eslint-disable-next-line
        const unreachable: never = message;
      }
    }
  }
});

function startTrainingOnceMnistLoads(message: StartTrainingRequest): void {
  const network = deserializeNetwork(message.networkBuffer);

  mnistProm.then((mnist) => {
    startTraining(network, message.hyperParams, mnist.training, mnist.test);
  });
}

function startTraining(
  network: Network,
  hyperParams: StochasticGradientDescentHyperParameters,
  trainingData: VectorLabeledImage[],
  testData: LabeledImage[]
): void {
  let epochsCompleted = 0;

  scheduleNextEpochUnlessTerminated();

  function scheduleNextEpochUnlessTerminated(): void {
    if (shouldTerminateAfterThisEpoch) {
      notifyListenersOfTermination(network);
    } else {
      requestAnimationFrame(performEpoch);
    }
  }

  function performEpoch(): void {
    network.stochasticGradientDescent(
      trainingData,
      hyperParams.batchSize,
      1,
      hyperParams.learningRate
    );

    notifyListenersOfEpochCompletion(network.test(testData), epochsCompleted);

    epochsCompleted++;
    if (epochsCompleted < hyperParams.epochs) {
      scheduleNextEpochUnlessTerminated();
    } else {
      trainedNetwork = option.some(network);
    }
  }
}

function notifyListenersOfTermination(network: Network): void {
  const message: TerminateTrainingResponse = {
    messageType: WorkerMessageType.TerminateTrainingResponse,
    networkBuffer: serializeNetwork(network),
  };
  self.postMessage(message, [message.networkBuffer]);
}

function notifyListenersOfEpochCompletion(
  accuracyRate: AccuracyRate,
  epoch: number
): void {
  const message: TrainingEpochCompleteNotification = {
    messageType: WorkerMessageType.TrainingEpochCompleteNotification,
    accuracyRate,
    epoch,
  };
  self.postMessage(message);
}

function terminateTrainingAfterThisEpoch(): void {
  trainedNetwork.match({
    none: () => {
      shouldTerminateAfterThisEpoch = true;
    },

    some: notifyListenersOfTermination,
  });
}
