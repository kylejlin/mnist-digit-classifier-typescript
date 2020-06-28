import { MatrixMap, Network } from ".";
import { Network1 } from "./networks/network1";

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
    return new Network1(sizes);
  },
  fromWeightsAndBiases(weights: MatrixMap, biases: MatrixMap): Network {
    return Network1.fromWeightsAndBiases(weights, biases);
  },
};
