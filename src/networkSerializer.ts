import { Network, MatrixMap } from "./network";
import { Matrix } from "./matrix";

export function serializeNetwork(network: Network): ArrayBuffer {
  const entries = getEntries(network);

  const numberOfBytesForSizes =
    (1 + network.sizes.length) * Uint32Array.BYTES_PER_ELEMENT;

  const buffer = new ArrayBuffer(
    numberOfBytesForSizes + entries.length * entries.BYTES_PER_ELEMENT
  );

  const uints = new Uint32Array(
    buffer,
    0,
    numberOfBytesForSizes / Uint32Array.BYTES_PER_ELEMENT
  );

  uints[0] = network.sizes.length;

  for (let i = 0; i < network.sizes.length; i++) {
    uints[1 + i] = network.sizes[i];
  }

  const floats = new Float64Array(buffer, numberOfBytesForSizes);
  floats.set(entries);

  return buffer;
}

function getEntries(network: Network): Float64Array {
  const weights = network.getWeights();
  const biases = network.getBiases();

  let entryCount = 0;

  for (let i = 1; i < weights.length; i++) {
    const weightMatrix = weights[i];
    const weightMatrixSize = weightMatrix.rows * weightMatrix.columns;
    entryCount += weightMatrixSize;

    const biasMatrix = biases[i];
    const biasMatrixSize = biasMatrix.rows * biasMatrix.columns;
    entryCount += biasMatrixSize;
  }

  const entries = new Float64Array(entryCount);

  let cursor = 0;
  for (let i = 1; i < weights.length; i++) {
    const weightMatrixEntries = weights[i].rowMajorOrderEntries();
    entries.set(weightMatrixEntries, cursor);
    cursor += weightMatrixEntries.length;

    const biasMatrixEntries = biases[i].rowMajorOrderEntries();
    entries.set(biasMatrixEntries, cursor);
    cursor += biasMatrixEntries.length;
  }

  return entries;
}

export function deserializeNetwork(buffer: ArrayBuffer): Network {
  const numberOfLayers = new Uint32Array(buffer, 0, 1)[0];
  const layerSizes = new Uint32Array(buffer, 4, numberOfLayers);
  const entries = new Float64Array(
    buffer.slice(Uint32Array.BYTES_PER_ELEMENT * (1 + numberOfLayers))
  );

  const weights: MatrixMap = [];
  const biases: MatrixMap = [];

  let cursor = 0;
  for (let i = 1; i < layerSizes.length; i++) {
    const outputLayerSize = layerSizes[i];
    const inputLayerSize = layerSizes[i - 1];

    {
      const rows = outputLayerSize;
      const columns = inputLayerSize;
      const size = rows * columns;
      weights[i] = Matrix.fromRowMajorOrderEntries(
        rows,
        columns,
        toArray(entries.subarray(cursor, cursor + size))
      );

      cursor += size;
    }

    {
      const rows = outputLayerSize;
      biases[i] = Matrix.fromRowMajorOrderEntries(
        rows,
        1,
        toArray(entries.subarray(cursor, cursor + rows))
      );

      cursor += rows;
    }
  }

  return Network.fromWeightsAndBiases(weights, biases);
}

function toArray(floats: Float64Array): number[] {
  const arr = new Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    arr[i] = floats[i];
  }
  return arr;
}
