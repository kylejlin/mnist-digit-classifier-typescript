import { Matrix } from "../matrix";

export interface LabeledImage {
  rows: number;
  columns: number;
  inputs: Matrix;
  label: number;
}

export interface VectorLabeledImage {
  rows: number;
  columns: number;
  inputs: Matrix;
  outputs: Matrix;
}

interface ImageMatrix {
  rows: number;
  columns: number;
  matrix: Matrix;
}

const Idx3FileFormat = {
  MagicNumber: { offset: 0, requiredValue: 0x00000803 },
  NumberOfImages: { offset: 4 },
  NumberOfRows: { offset: 8 },
  NumberOfColumns: { offset: 12 },
} as const;

const Idx1FileFormat = {
  MagicNumber: { offset: 0, requiredValue: 0x00000801 },
  NumberOfLabels: { offset: 4 },
} as const;

export function getLabeledImages(
  imagesBuffer: ArrayBuffer,
  labelsBuffer: ArrayBuffer
): LabeledImage[] {
  const images = getImages(imagesBuffer);
  const labels = getLabels(labelsBuffer);

  if (images.length !== labels.length) {
    throw new Error(
      "There are " +
        images.length +
        " images, but " +
        labels.length +
        " labels. There must be the same amount of images and labels."
    );
  }

  const labeledImages: LabeledImage[] = new Array(images.length);
  for (let i = 0; i < images.length; i++) {
    const { rows, columns, matrix } = images[i];
    labeledImages[i] = { rows, columns, inputs: matrix, label: labels[i] };
  }
  return labeledImages;
}

function getImages(buffer: ArrayBuffer): ImageMatrix[] {
  const bytes = new Uint8Array(buffer);

  assertIdx3MagicNumberIsCorrect(bytes);

  const numberOfImages = getInt32MsbFirst(
    bytes,
    Idx3FileFormat.NumberOfImages.offset
  );
  const rows = getInt32MsbFirst(bytes, Idx3FileFormat.NumberOfRows.offset);
  const columns = getInt32MsbFirst(
    bytes,
    Idx3FileFormat.NumberOfColumns.offset
  );
  const size = rows * columns;

  let imagesParsed = 0;
  const images: ImageMatrix[] = new Array(numberOfImages);
  const firstPixelIndex = Idx3FileFormat.NumberOfColumns.offset + 4;

  while (imagesParsed < numberOfImages) {
    const vectorEntries: number[] = new Array(size);
    for (let j = 0; j < size; j++) {
      vectorEntries[j] = bytes[firstPixelIndex + imagesParsed * size + j] / 255;
    }

    images[imagesParsed] = {
      rows,
      columns,
      matrix: Matrix.columnVector(vectorEntries),
    };
    imagesParsed++;
  }

  return images;
}

function assertIdx3MagicNumberIsCorrect(bytes: Uint8Array): void {
  const actual = getInt32MsbFirst(bytes, Idx3FileFormat.MagicNumber.offset);
  const expected = Idx3FileFormat.MagicNumber.requiredValue;
  if (actual !== expected) {
    throw new Error(
      "The first 4 bytes of an idx3 file must be 0x" +
        expected.toString(16) +
        ", but the first 4 bytes of the provided file were 0x" +
        actual.toString(16)
    );
  }
}

function getInt32MsbFirst(bytes: Uint8Array, offset: number): number {
  return (
    (bytes[offset] << 24) |
    (bytes[offset + 1] << 16) |
    (bytes[offset + 2] << 8) |
    bytes[offset + 3]
  );
}

function getLabels(buffer: ArrayBuffer): number[] {
  const bytes = new Uint8Array(buffer);

  assertIdx1MagicNumberIsCorrect(bytes);

  const numberOfLabels = getInt32MsbFirst(
    bytes,
    Idx1FileFormat.NumberOfLabels.offset
  );
  const labels: number[] = new Array(numberOfLabels);
  const firstLabelIndex = Idx1FileFormat.NumberOfLabels.offset + 4;
  for (let i = 0; i < numberOfLabels; i++) {
    labels[i] = bytes[firstLabelIndex + i];
  }
  return labels;
}

function assertIdx1MagicNumberIsCorrect(bytes: Uint8Array): void {
  const actual = getInt32MsbFirst(bytes, Idx1FileFormat.MagicNumber.offset);
  const expected = Idx1FileFormat.MagicNumber.requiredValue;
  if (actual !== expected) {
    throw new Error(
      "The first 4 bytes of an idx1 file must be 0x" +
        expected.toString(16) +
        ", but the first 4 bytes of the provided file were 0x" +
        actual.toString(16)
    );
  }
}

export function convertLabelToVector(image: LabeledImage): VectorLabeledImage {
  const entries: number[] = new Array(10).fill(0);
  entries[image.label] = 1;
  const outputs = Matrix.columnVector(entries);
  return {
    rows: image.rows,
    columns: image.columns,
    inputs: image.inputs,
    outputs,
  };
}
