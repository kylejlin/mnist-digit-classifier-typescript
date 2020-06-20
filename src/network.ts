import { LabeledImage, VectorLabeledImage } from "./data";
import { Matrix } from "./matrix";

export class Network {
  private layers: number;
  private weights: MatrixMap;
  private biases: MatrixMap;
  private log: (message: string) => void;

  constructor(sizes: number[], log?: (message: string) => void) {
    this.layers = sizes.length;

    this.weights = [];
    this.biases = [];
    for (let outputLayer = 1; outputLayer < sizes.length; outputLayer++) {
      const inputLayer = outputLayer - 1;
      const outputLayerSize = sizes[outputLayer];
      const inputLayerSize = sizes[inputLayer];
      this.weights[outputLayer] = Matrix.random(
        outputLayerSize,
        inputLayerSize
      );
      this.biases[outputLayer] = Matrix.random(outputLayerSize, 1);
    }

    this.log = log || (() => {});
  }

  stochasticGradientDescent(
    trainingData: VectorLabeledImage[],
    miniBatchSize: number,
    epochs: number,
    learningRate: number,
    testData?: LabeledImage[]
  ): void {
    for (let epoch = 0; epoch < epochs; epoch++) {
      const miniBatches = divideIntoMiniBatches(trainingData, miniBatchSize);
      for (const miniBatch of miniBatches) {
        const { weightGradients, biasGradients } = this.getAverageGradients(
          miniBatch
        );

        for (let i = 1; i < this.layers; i++) {
          weightGradients[i].mutMultiplyScalar(learningRate);
          biasGradients[i].mutMultiplyScalar(learningRate);

          this.weights[i].mutAdd(weightGradients[i]);
          this.biases[i].mutAdd(biasGradients[i]);
        }
      }

      if (testData !== undefined) {
        const numCorrect = this.getCorrectClassifications(testData);
        this.log(
          "Epoch " +
            epoch +
            ": " +
            numCorrect +
            " / " +
            testData.length +
            " correct"
        );
      }
    }
  }

  private getAverageGradients(miniBatch: VectorLabeledImage[]): Gradients {
    const weightGradients = this.getZeroMatricesForWeightGradients();
    const biasGradients = this.getZeroMatricesForBiasGradients();

    for (const image of miniBatch) {
      const imageGradients = this.getGradients(image);

      for (let i = 1; i < this.layers; i++) {
        weightGradients[i].mutAdd(imageGradients.weightGradients[i]);
        biasGradients[i].mutAdd(imageGradients.biasGradients[i]);
      }
    }

    for (let i = 1; i < this.layers; i++) {
      weightGradients[i].mutMultiplyScalar(1 / miniBatch.length);
      biasGradients[i].mutMultiplyScalar(1 / miniBatch.length);
    }

    return { weightGradients, biasGradients };
  }

  private getZeroMatricesForWeightGradients(): MatrixMap {
    const matrices: MatrixMap = [];
    for (let i = 1; i < this.layers; i++) {
      const weightMatrix = this.weights[i];
      matrices[i] = Matrix.zeros(weightMatrix.rows, weightMatrix.columns);
    }
    return matrices;
  }

  private getZeroMatricesForBiasGradients(): MatrixMap {
    const matrices: MatrixMap = [];
    for (let i = 1; i < this.layers; i++) {
      const biasMatrix = this.biases[i];
      matrices[i] = Matrix.zeros(biasMatrix.rows, biasMatrix.columns);
    }
    return matrices;
  }

  private getGradients(image: VectorLabeledImage): Gradients {
    const { weightedSums, activations } = this.performForwardPass(image.inputs);
    const errors: MatrixMap = [];
    const weightGradients: MatrixMap = [];
    const biasGradients: MatrixMap = [];

    const lastLayerError = weightedSums[this.layers - 1].immutApplyElementwise(
      sigmaPrime
    );
    lastLayerError.mutHadamard(
      this.getLastLayerCostDerivative(
        activations[this.layers - 1],
        image.outputs
      )
    );
    errors[this.layers - 1] = lastLayerError;

    for (let i = this.layers - 2; i >= 1; i--) {
      const error = this.weights[i + 1]
        .immutTranspose()
        .immutMultiply(errors[i + 1]);
      error.mutHadamard(weightedSums[i].immutApplyElementwise(sigmaPrime));
      errors[i] = error;

      weightGradients[i] = error.immutMultiply(
        activations[i - 1].immutTranspose()
      );

      biasGradients[i] = error;
    }

    return { weightGradients, biasGradients };
  }

  private performForwardPass(inputs: Matrix): WeightedSumsAndActivations {
    const weightedSums: MatrixMap = [];
    const activations: MatrixMap = [inputs];

    for (let outputLayer = 1; outputLayer < this.layers; outputLayer++) {
      const inputLayer = outputLayer - 1;
      const weightedSum = this.weights[outputLayer].immutMultiply(
        activations[inputLayer]
      );
      weightedSum.mutAdd(this.biases[outputLayer]);
      weightedSums[outputLayer] = weightedSum;
      activations[outputLayer] = weightedSum.immutApplyElementwise(sigma);
    }
    return { weightedSums, activations };
  }

  private getLastLayerCostDerivative(
    actualOutput: Matrix,
    expectedOutput: Matrix
  ): Matrix {
    return actualOutput.immutSubtract(expectedOutput);
  }

  private getCorrectClassifications(testData: LabeledImage[]): number {
    let correctClassifications = 0;
    for (const image of testData) {
      const { activations } = this.performForwardPass(image.inputs);
      const prediction = argmax(
        activations[this.layers - 1].rowMajorOrderEntries()
      );
      if (prediction === image.label) {
        correctClassifications++;
      }
    }
    return correctClassifications;
  }
}

export interface MatrixMap {
  [layer: number]: Matrix;
}

export interface Gradients {
  weightGradients: MatrixMap;
  biasGradients: MatrixMap;
}

export interface WeightedSumsAndActivations {
  weightedSums: MatrixMap;
  activations: MatrixMap;
}

function divideIntoMiniBatches(
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

function sigma(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

function sigmaPrime(z: number): number {
  const sigmaZ = sigma(z);
  return sigmaZ * (1 - sigmaZ);
}

function argmax(arr: readonly number[]): number {
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
