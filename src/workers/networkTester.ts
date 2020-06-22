import { AccuracyRate, LabeledImage } from "../data";
import { mnistProm } from "../data/mnist";
import { Network } from "../network";
import { deserializeNetwork } from "../networkSerializer";
import {
  StartTestingRequest,
  TestCompleteNotification,
  WorkerMessageType,
} from "../workerMessages";

interface Self {
  postMessage(
    message: TestCompleteNotification,
    transfers?: Transferable[]
  ): void;
  addEventListener: Worker["addEventListener"];
}

declare const self: Self;

export {};

self.addEventListener("message", (event) => {
  const { data } = event;

  if (data !== null && "object" === typeof data && "messageType" in data) {
    const message: StartTestingRequest = data;
    startTestingOnceMnistLoads(message);
  }
});

function startTestingOnceMnistLoads(message: StartTestingRequest): void {
  const network = deserializeNetwork(message.networkBuffer);
  mnistProm.then((mnist) => {
    startTesting(network, mnist.test);
  });
}

function startTesting(network: Network, testData: LabeledImage[]): void {
  const accuracyRate = network.test(testData);
  notifyListenersOfTestCompletion(accuracyRate);
}

function notifyListenersOfTestCompletion(accuracyRate: AccuracyRate): void {
  const message: TestCompleteNotification = {
    messageType: WorkerMessageType.TestCompleteNotification,
    accuracyRate,
  };
  self.postMessage(message);
}
