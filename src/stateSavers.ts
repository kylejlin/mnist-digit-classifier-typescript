import { Option, option } from "rusty-ts";
import { Matrix } from "./matrix";
import { Network } from "./network";
import { CustomImage } from "./state";
import { deserializeNetwork, serializeNetwork } from "./networkSerializer";

export interface StateSaver<T> {
  getState(): Option<T>;
  saveState(state: T): void;
}

enum LocalStorageKeys {
  CustomImages = "CustomImages",
  NeuralNetwork = "NeuralNetwork",
}

// 28*28 pixels plus 1 label
const BYTES_PER_IMAGE = 28 * 28 + 1;

export const imageSaver: StateSaver<CustomImage[]> = {
  getState(): Option<CustomImage[]> {
    const stateStr = localStorage.getItem(LocalStorageKeys.CustomImages);
    if (stateStr === null) {
      return option.none();
    } else {
      const bytes = decodeBytes(stateStr);

      if (bytes.length % BYTES_PER_IMAGE !== 0) {
        throw new Error(
          "Cannot decode image bytes because there are " +
            bytes.length +
            " bytes, and that number is not a multiple of 784."
        );
      }

      const numberOfImages = bytes.length / BYTES_PER_IMAGE;

      const byteVectors: CustomImage[] = new Array(numberOfImages);
      for (let i = 0; i < numberOfImages; i++) {
        const imageBytes = bytes.subarray(
          i * BYTES_PER_IMAGE,
          i * BYTES_PER_IMAGE + 784
        );
        const u8Matrix = Matrix.columnVector(toArray(imageBytes));
        byteVectors[i] = { u8Matrix, label: bytes[i * BYTES_PER_IMAGE + 784] };
      }
      return option.some(byteVectors);
    }
  },

  saveState(images: CustomImage[]): void {
    const bytes = new Uint8Array(images.length * BYTES_PER_IMAGE);
    for (let i = 0; i < images.length; i++) {
      const image = images[i];
      bytes.set(image.u8Matrix.rowMajorOrderEntries(), i * BYTES_PER_IMAGE);
      bytes[i * BYTES_PER_IMAGE + 784] = image.label;
    }

    const stateStr = encodeBytes(bytes);
    localStorage.setItem(LocalStorageKeys.CustomImages, stateStr);
  },
};

export const networkSaver: StateSaver<Network> = {
  getState(): Option<Network> {
    const stateStr = localStorage.getItem(LocalStorageKeys.NeuralNetwork);
    if (stateStr === null) {
      return option.none();
    } else {
      const bytes = decodeBytes(stateStr);
      const network = deserializeNetwork(bytes.buffer);
      return option.some(network);
    }
  },

  saveState(network: Network): void {
    const buffer = serializeNetwork(network);
    const stateStr = encodeBytes(new Uint8Array(buffer));
    localStorage.setItem(LocalStorageKeys.NeuralNetwork, stateStr);
  },
};

function decodeBytes(str: string): Uint8Array {
  const numberOfBytes = toU32((str.charCodeAt(0) << 16) | str.charCodeAt(1));
  const bytes = new Uint8Array(numberOfBytes);

  for (let i = 0; i < numberOfBytes; i++) {
    const code = str.charCodeAt(2 + Math.floor(i / 2));

    const byte = i % 2 === 0 ? code >>> 8 : code;

    bytes[i] = byte;
  }
  return bytes;
}

function toU32(n: number): number {
  const arr = new Uint32Array(1);
  arr[0] = n;
  return arr[0];
}

function toArray(u8s: Uint8Array): number[] {
  const arr: number[] = new Array(u8s.length);
  for (let i = 0; i < u8s.length; i++) {
    arr[i] = u8s[i];
  }
  return arr;
}

function encodeBytes(bytes: Uint8Array): string {
  const numberOfBytes = bytes.length;
  const u16s = new Array(Math.ceil(numberOfBytes / 2));

  for (let i = 0; i < bytes.length; i++) {
    const byte = bytes[i];
    if (i % 2 === 0) {
      u16s[i / 2] = byte << 8;
    } else {
      u16s[(i - 1) / 2] |= byte;
    }
  }

  return (
    String.fromCharCode(
      numberOfBytes >>> 16,
      numberOfBytes & 0b0000_0000_0000_0000_1111_1111_1111_1111
    ) + stringifyU16s(u16s)
  );
}

function stringifyU16s(u16s: number[]): string {
  try {
    // This will crash in some browsers if
    // `u16s` is too large.
    return String.fromCharCode(...u16s);
  } catch {
    let out = "";

    for (let i = 0; i < u16s.length; i++) {
      out += String.fromCharCode(u16s[i]);
    }

    return out;
  }
}
