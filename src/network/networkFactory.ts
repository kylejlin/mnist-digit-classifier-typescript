import { MatrixMap, Network } from ".";
import { Chapter3CrossEntropyL2Network } from "./networks/ch3crossEntropyL2";

export interface NetworkFactory {
  fromSizes(sizes: number[]): Network;
  fromWeightsAndBiases(weights: MatrixMap, biases: MatrixMap): Network;
}

/**
 * At any given time, the entire web app uses one and only
 * one neural network implementation.
 *
 * All other files will depend on `networkFactory`, which will
 * allow me to easily change which network I'm using by changing
 * only one variable.
 * Otherwise, if I wanted to change from using `NetworkX`
 * to `NetworkY`, I would have to go through the entire src
 * folder and replace every occurrence of `NetworkX` with
 * `NetworkY`.
 */
export const networkFactory: NetworkFactory = {
  fromSizes(sizes: number[]): Network {
    return new Chapter3CrossEntropyL2Network(sizes);
  },
  fromWeightsAndBiases(weights: MatrixMap, biases: MatrixMap): Network {
    return Chapter3CrossEntropyL2Network.fromWeightsAndBiases(weights, biases);
  },
};
