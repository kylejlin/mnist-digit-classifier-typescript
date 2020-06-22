import { NetworkTrainerRequest } from "../networkServices";

export default class NetworkTrainerWorker {
  postMessage(message: NetworkTrainerRequest, transfers?: Transferable[]): void;
  addEventListener: Worker["addEventListener"];
  terminate: Worker["terminate"];
}
