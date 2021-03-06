import { Matrix } from "./matrix";
import { MatrixMap, Network } from "./network";
import { networkFactory } from "./network/networkFactory";

export function serializeNetwork(network: Network): ArrayBuffer {
  const entries = getEntries(network);

  const numberOfBytesForSizes =
    (1 + network.layerSizes.length) * Uint32Array.BYTES_PER_ELEMENT;

  const buffer = new ArrayBuffer(
    numberOfBytesForSizes + entries.length * entries.BYTES_PER_ELEMENT
  );

  const uints = new Uint32Array(
    buffer,
    0,
    numberOfBytesForSizes / Uint32Array.BYTES_PER_ELEMENT
  );

  uints[0] = network.layerSizes.length;

  for (let i = 0; i < network.layerSizes.length; i++) {
    uints[1 + i] = network.layerSizes[i];
  }

  const floats = new Float64Array(entries);
  putBuffer(floats.buffer, buffer, numberOfBytesForSizes);

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

function putBuffer(
  src: ArrayBuffer,
  dest: ArrayBuffer,
  byteOffset: number = 0
): void {
  const srcU8s = new Uint8Array(src);
  const destU8s = new Uint8Array(dest);
  for (let i = 0; i < srcU8s.length; i++) {
    destU8s[byteOffset + i] = srcU8s[i];
  }
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

  return networkFactory.fromWeightsAndBiases(weights, biases);
}

function toArray(floats: Float64Array): number[] {
  const arr = new Array(floats.length);
  for (let i = 0; i < floats.length; i++) {
    arr[i] = floats[i];
  }
  return arr;
}
