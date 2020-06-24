import url from "url";
import {
  convertLabelToVector,
  getLabeledImages,
  LabeledImage,
  VectorLabeledImage,
} from ".";

export interface MnistData {
  training: VectorLabeledImage[];
  test: LabeledImage[];
}

declare const self: { location: Window["location"] };

const { location } = self;

const TRAINING_IMAGES_URL = url.resolve(
  location.href,
  "./assets/train60k-images-idx3-ubyte"
);
const TRAINING_LABELS_URL = url.resolve(
  location.href,
  "./assets/train60k-labels-idx1-ubyte"
);
const TEST_IMAGES_URL = url.resolve(
  location.href,
  "./assets/test10k-images-idx3-ubyte"
);
const TEST_LABELS_URL = url.resolve(
  location.href,
  "./assets/test10k-labels-idx1-ubyte"
);
const trainingImagesProm: Promise<ArrayBuffer> = getArrayBuffer(
  TRAINING_IMAGES_URL
);
const trainingLabelsProm: Promise<ArrayBuffer> = getArrayBuffer(
  TRAINING_LABELS_URL
);
const testImagesProm: Promise<ArrayBuffer> = getArrayBuffer(TEST_IMAGES_URL);
const testLabelsProm: Promise<ArrayBuffer> = getArrayBuffer(TEST_LABELS_URL);

function getArrayBuffer(url: string): Promise<ArrayBuffer> {
  return fetch(url).then((response) => {
    if (200 <= response.status && response.status <= 299) {
      return response.arrayBuffer();
    } else {
      return getErrorMessage(response).then((errorMessage) =>
        Promise.reject(
          new Error(
            "Tried to fetch " +
              url +
              " but got the following error: " +
              errorMessage
          )
        )
      );
    }
  });
}

function getErrorMessage(response: Response): Promise<string> {
  const { status, statusText } = response;
  return response
    .text()
    .then((text) => status + " (" + statusText + "): " + text);
}

export const mnistProm: Promise<MnistData> = Promise.all([
  trainingImagesProm,
  trainingLabelsProm,
  testImagesProm,
  testLabelsProm,
]).then(
  ([
    trainingImagesBuffer,
    trainingLabelsBuffer,
    testImagesBuffer,
    testLabelsBuffer,
  ]) => {
    return {
      training: getLabeledImages(
        trainingImagesBuffer,
        trainingLabelsBuffer
      ).map(convertLabelToVector),
      test: getLabeledImages(testImagesBuffer, testLabelsBuffer),
    };
  }
);
