import {
  MatrixMap,
  Network,
  StochasticGradientDescentHyperParameters,
  WeightedSumsAndActivations,
} from "..";
import { AccuracyRate, LabeledImage, VectorLabeledImage } from "../../data";
import { DeepReadonly } from "../../deepReadonly";
import { Matrix } from "../../matrix";
import { argmax, divideIntoMiniBatches, Gradients } from "../utils";

export class Network1 implements Network {
  private layers: number;
  private weights: MatrixMap;
  private biases: MatrixMap;
  private log: (accuracyRate: AccuracyRate, epoch: number) => void;

  public readonly sizes: number[];

  static fromWeightsAndBiases(weights: MatrixMap, biases: MatrixMap): Network {
    const sizes = [weights[1].columns];
    for (let i = 1; i < weights.length; i++) {
      sizes.push(weights[i].rows);
    }

    const network = new Network1(sizes);

    for (let i = 1; i < weights.length; i++) {
      network.weights[i] = weights[i];
      network.biases[i] = biases[i];
    }

    return network;
  }

  constructor(
    sizes: number[],
    log?: (accuracyRate: AccuracyRate, epoch: number) => void
  ) {
    this.layers = sizes.length;

    this.weights = [];
    this.biases = [];
    for (let outputLayer = 1; outputLayer < sizes.length; outputLayer++) {
      const inputLayer = outputLayer - 1;
      const outputLayerSize = sizes[outputLayer];
      const inputLayerSize = sizes[inputLayer];
      this.weights[outputLayer] = Matrix.randomUniform(
        outputLayerSize,
        inputLayerSize
      );
      this.biases[outputLayer] = Matrix.randomUniform(outputLayerSize, 1);
    }

    this.log = log || (() => {});

    this.sizes = sizes;
  }

  stochasticGradientDescent(
    trainingData: VectorLabeledImage[],
    hyperparams: StochasticGradientDescentHyperParameters,
    testData?: LabeledImage[]
  ): void {
    const { batchSize, epochs, learningRate } = hyperparams;

    for (let epoch = 0; epoch < epochs; epoch++) {
      const miniBatches = divideIntoMiniBatches(trainingData, batchSize);
      for (const miniBatch of miniBatches) {
        const { weightGradients, biasGradients } = this.getAverageGradients(
          miniBatch
        );

        for (let i = 1; i < this.layers; i++) {
          weightGradients[i].mutMultiplyScalar(learningRate);
          biasGradients[i].mutMultiplyScalar(learningRate);

          this.weights[i].mutSubtract(weightGradients[i]);
          this.biases[i].mutSubtract(biasGradients[i]);
        }
      }

      if (testData !== undefined) {
        const accuracyRate = this.test(testData);
        this.log(accuracyRate, epoch);
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
    weightGradients[this.layers - 1] = lastLayerError.immutMultiply(
      activations[this.layers - 2].immutTranspose()
    );
    biasGradients[this.layers - 1] = lastLayerError;

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

  performForwardPass(inputs: Matrix): WeightedSumsAndActivations {
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

  test(testData: LabeledImage[]): AccuracyRate {
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
    return { correct: correctClassifications, total: testData.length };
  }

  getWeights(): DeepReadonly<MatrixMap> {
    return this.weights;
  }

  getBiases(): DeepReadonly<MatrixMap> {
    return this.biases;
  }
}

function sigma(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

function sigmaPrime(z: number): number {
  const sigmaZ = sigma(z);
  return sigmaZ * (1 - sigmaZ);
}
