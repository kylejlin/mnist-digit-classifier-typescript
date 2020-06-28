import { MatrixMap } from ".";
import { VectorLabeledImage } from "../data";

export interface Gradients {
  weightGradients: MatrixMap;
  biasGradients: MatrixMap;
}

export function divideIntoMiniBatches(
  trainingData: VectorLabeledImage[],
  miniBatchSize: number
): VectorLabeledImage[][] {
  shuffle(trainingData);
  const miniBatches: VectorLabeledImage[][] = [];
  for (let i = 0; i < trainingData.length; i += miniBatchSize) {
    miniBatches.push(trainingData.slice(i, i + miniBatchSize));
  }
  return miniBatches;
}

function shuffle(arr: unknown[]): void {
  const SHUFFLE_TIMES = 512;

  for (let n = 0; n < SHUFFLE_TIMES; n++) {
    for (let i = arr.length - 1; i >= 1; i--) {
      let j = randInt(i + 1);
      const temp = arr[i];
      arr[i] = arr[j];
      arr[j] = temp;
    }
  }
}

function randInt(exclMax: number): number {
  return Math.floor(Math.random() * exclMax);
}

export function argmax(arr: readonly number[]): number {
  let maxIndex = 0;
  let max = arr[maxIndex];
  for (let i = 1; i < arr.length; i++) {
    const value = arr[i];
    if (value > max) {
      max = value;
      maxIndex = i;
    }
  }
  return maxIndex;
}
