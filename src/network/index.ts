import { LabeledImage, VectorLabeledImage, AccuracyRate } from "../data";
import { Matrix } from "../matrix";
import { DeepReadonly } from "../deepReadonly";

export interface Network {
  readonly sizes: number[];

  stochasticGradientDescent(
    trainingData: VectorLabeledImage[],
    hyperparams: StochasticGradientDescentHyperParameters,
    evaluationData?: LabeledImage[]
  ): void;

  performForwardPass(inputColumnVector: Matrix): WeightedSumsAndActivations;

  test(testData: LabeledImage[]): AccuracyRate;

  getWeights(): DeepReadonly<MatrixMap>;

  getBiases(): DeepReadonly<MatrixMap>;
}

export interface WeightedSumsAndActivations {
  weightedSums: MatrixMap;
  activations: MatrixMap;
}

export interface MatrixMap {
  [layer: number]: Matrix;
  length: number;
}

export interface StochasticGradientDescentHyperParameters {
  batchSize: number;
  epochs: number;
  learningRate: number;
  regularizationRate: number;
}
