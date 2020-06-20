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

const TRAINING_IMAGES_URL = url.resolve(
  window.location.href,
  "./assets/train60k-images-idx3-ubyte"
);
const TRAINING_LABELS_URL = url.resolve(
  window.location.href,
  "./assets/train60k-labels-idx1-ubyte"
);
const TEST_IMAGES_URL = url.resolve(
  window.location.href,
  "./assets/test10k-images-idx3-ubyte"
);
const TEST_LABELS_URL = url.resolve(
  window.location.href,
  "./assets/test10k-labels-idx1-ubyte"
);
const trainingImagesProm: Promise<ArrayBuffer> = fetch(
  TRAINING_IMAGES_URL
).then((response) => response.arrayBuffer());
const trainingLabelsProm: Promise<ArrayBuffer> = fetch(
  TRAINING_LABELS_URL
).then((response) => response.arrayBuffer());
const testImagesProm: Promise<ArrayBuffer> = fetch(
  TEST_IMAGES_URL
).then((response) => response.arrayBuffer());
const testLabelsProm: Promise<ArrayBuffer> = fetch(
  TEST_LABELS_URL
).then((response) => response.arrayBuffer());

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
