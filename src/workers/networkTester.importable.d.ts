import { StartTestingRequest } from "../networkServices";

export default class NetworkTesterWorker {
  postMessage(message: StartTestingRequest, transfers?: Transferable[]): void;
  addEventListener: Worker["addEventListener"];
  terminate: Worker["terminate"];
}
