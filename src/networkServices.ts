import { AccuracyRate } from "./data";
import { Network } from "./network";
import { deserializeNetwork, serializeNetwork } from "./networkSerializer";
import {
  NetworkTrainerNotification,
  StartTestingRequest,
  StartTrainingRequest,
  StochasticGradientDescentHyperParameters,
  TerminateTrainingRequest,
  TerminateTrainingResponse,
  TestCompleteNotification,
  TrainingEpochCompleteNotification,
  WorkerMessageType,
} from "./workerMessages";
import NetworkTesterWorker from "./workers/networkTester.importable";
import NetworkTrainerWorker from "./workers/networkTrainer.importable";

export interface NetworkTrainer {
  start(): void;
  terminate(): void;
}

export interface NetworkTester {
  start(): void;
  terminate(): void;
}

export interface NetworkTrainerListeners {
  onEpochComplete(accuracyRate: AccuracyRate, epoch: number): void;
  onTerminate(network: Network): void;
}

export interface NetworkTesterListeners {
  onComplete(rate: AccuracyRate): void;
}

export function trainNetwork(
  network: Network,
  hyperParams: StochasticGradientDescentHyperParameters,
  listeners: NetworkTrainerListeners
): NetworkTrainer {
  const worker = new NetworkTrainerWorker();

  worker.addEventListener("message", (event) => {
    const { data } = event;
    if (data !== null && "object" === typeof data && "messageType" in data) {
      const message: NetworkTrainerNotification = data;
      switch (message.messageType) {
        case WorkerMessageType.TrainingEpochCompleteNotification:
          notifyListenersOfEpochCompletion(message);
          break;
        case WorkerMessageType.TerminateTrainingResponse:
          notifyListenersOfTermination(message);
          break;

        default: {
          // Force exhaustive matching

          // eslint-disable-next-line
          const unreachable: never = message;
        }
      }
    }
  });

  return { start: startTraining, terminate: stopTrainingAfterThisEpoch };

  function notifyListenersOfEpochCompletion(
    message: TrainingEpochCompleteNotification
  ): void {
    listeners.onEpochComplete(message.accuracyRate, message.epoch);
  }

  function notifyListenersOfTermination(
    message: TerminateTrainingResponse
  ): void {
    const updatedNetwork = deserializeNetwork(message.networkBuffer);
    listeners.onTerminate(updatedNetwork);

    worker.terminate();
  }

  function startTraining(): void {
    const message: StartTrainingRequest = {
      messageType: WorkerMessageType.StartTrainingRequest,
      networkBuffer: serializeNetwork(network),
      hyperParams,
    };
    worker.postMessage(message, [message.networkBuffer]);
  }

  function stopTrainingAfterThisEpoch(): void {
    const message: TerminateTrainingRequest = {
      messageType: WorkerMessageType.TerminateTrainingRequest,
    };
    worker.postMessage(message);
  }
}

export function testNetwork(
  network: Network,
  listeners: NetworkTesterListeners
): NetworkTrainer {
  const worker = new NetworkTesterWorker();

  worker.addEventListener("message", (event) => {
    const { data } = event;
    if (data !== null && "object" === typeof data && "messageType" in data) {
      const message: TestCompleteNotification = data;
      notifyListenersOfTestingCompletion(message);
    }
  });

  return { start: startTesting, terminate: terminateWorker };

  function notifyListenersOfTestingCompletion(
    message: TestCompleteNotification
  ): void {
    listeners.onComplete(message.accuracyRate);

    worker.terminate();
  }

  function startTesting(): void {
    const message: StartTestingRequest = {
      messageType: WorkerMessageType.StartTestingRequest,
      networkBuffer: serializeNetwork(network),
    };
    worker.postMessage(message, [message.networkBuffer]);
  }

  function terminateWorker(): void {
    worker.terminate();
  }
}
