export class Matrix {
  static randomUniform(rows: number, columns: number): Matrix {
    const size = rows * columns;
    const data = new Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.random() * 2 - 1;
    }
    return new Matrix(rows, columns, data);
  }

  static zeros(rows: number, columns: number): Matrix {
    const data = new Array(rows * columns).fill(0);
    return new Matrix(rows, columns, data);
  }

  static fromRows(rows: number[][]): Matrix {
    const columns = rows[0].length;
    if (rows.some((row) => row.length !== columns)) {
      throw new Error(
        "Cannot create a matrix from a jagged array: " + JSON.stringify(rows)
      );
    }

    return new Matrix(rows.length, columns, rows.flat());
  }

  static columnVector(entries: number[]): Matrix {
    return new Matrix(entries.length, 1, entries);
  }

  public readonly rows: number;
  public readonly columns: number;
  private data: number[];

  private constructor(rows: number, columns: number, data: number[]) {
    this.rows = rows;
    this.columns = columns;
    this.data = data;
  }

  clone(): Matrix {
    return new Matrix(this.rows, this.columns, this.data.slice());
  }

  mutMultiplyScalar(n: number): void {
    const size = this.data.length;
    for (let i = 0; i < size; i++) {
      this.data[i] *= n;
    }
  }

  mutAdd(other: Matrix): void {
    if (!(other.rows === this.rows && other.columns === this.columns)) {
      throw new TypeError(
        "Cannot add a " +
          this.rows +
          "x" +
          this.columns +
          " to a " +
          other.rows +
          "x" +
          other.columns +
          " matrix."
      );
    }

    const size = this.data.length;
    for (let i = 0; i < size; i++) {
      this.data[i] += other.data[i];
    }
  }

  mutSubtract(other: Matrix): void {
    if (!(other.rows === this.rows && other.columns === this.columns)) {
      throw new TypeError(
        "Cannot add a " +
          this.rows +
          "x" +
          this.columns +
          " to a " +
          other.rows +
          "x" +
          other.columns +
          " matrix."
      );
    }

    const size = this.data.length;
    for (let i = 0; i < size; i++) {
      this.data[i] -= other.data[i];
    }
  }

  immutSubtract(other: Matrix): Matrix {
    if (!(other.rows === this.rows && other.columns === this.columns)) {
      throw new TypeError(
        "Cannot add a " +
          this.rows +
          "x" +
          this.columns +
          " matrix to a " +
          other.rows +
          "x" +
          other.columns +
          " matrix."
      );
    }

    const clone = this.clone();
    const size = clone.data.length;
    for (let i = 0; i < size; i++) {
      clone.data[i] -= other.data[i];
    }
    return clone;
  }

  immutMultiply(other: Matrix): Matrix {
    if (this.columns !== other.rows) {
      throw new TypeError(
        "Cannot multiply a " +
          this.rows +
          "x" +
          this.columns +
          " matrix with a " +
          other.rows +
          "x" +
          other.columns +
          " matrix."
      );
    }

    const product = Matrix.zeros(this.rows, other.columns);

    const thisData = this.data;
    const otherData = other.data;
    const productData = product.data;
    const thisRows = this.rows;
    const otherColumns = other.columns;
    const thisColumns = this.columns;
    const productColumns = product.columns;

    for (let thisR = 0; thisR < thisRows; thisR++) {
      for (let otherC = 0; otherC < otherColumns; otherC++) {
        for (let thisC = 0; thisC < thisColumns; thisC++) {
          productData[thisR * productColumns + otherC] +=
            thisData[thisR * thisColumns + thisC] *
            otherData[thisC * otherColumns + otherC];
        }
      }
    }
    return product;
  }

  mutHadamard(other: Matrix): void {
    if (!(other.rows === this.rows && other.columns === this.columns)) {
      throw new TypeError(
        "Cannot take the Hadamard product of a " +
          this.rows +
          "x" +
          this.columns +
          " matrix and a " +
          other.rows +
          "x" +
          other.columns +
          " matrix."
      );
    }

    const size = this.data.length;
    for (let i = 0; i < size; i++) {
      this.data[i] *= other.data[i];
    }
  }

  immutTranspose(): Matrix {
    const transposed = new Matrix(this.columns, this.rows, this.data.slice());
    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.columns; c++) {
        transposed.data[c * transposed.columns + r] = this.data[
          r * this.columns + c
        ];
      }
    }
    return transposed;
  }

  rowMajorOrderEntries(): readonly number[] {
    return this.data;
  }

  immutApplyElementwise(f: (entry: number) => number): Matrix {
    const clone = this.clone();
    const cloneData = clone.data;
    const size = cloneData.length;
    for (let i = 0; i < size; i++) {
      cloneData[i] = f(cloneData[i]);
    }
    return clone;
  }

  print(decimals: number): string {
    const entries = this.rowMajorOrderEntries();
    const entryStrings = entries.map((entry) => entry.toFixed(decimals));
    const entryStringLengths = entryStrings.map((s) => s.length);
    const maxLength = Math.max(...entryStringLengths);

    const topAndBottomBorder = "-".repeat(
      this.columns * (maxLength + " | ".length) - " | ".length
    );

    let str = topAndBottomBorder + "\n";

    for (let r = 0; r < this.rows; r++) {
      for (let c = 0; c < this.columns; c++) {
        str +=
          leftpad(entryStrings[r * this.columns + c], maxLength, " ") + " | ";
      }

      str = str.slice(0, -" | ".length);

      str += "\n";
    }

    str += topAndBottomBorder;
    return str;
  }
}

function leftpad(s: string, minLength: number, fillCharacter: string): string {
  const diff = minLength - s.length;
  if (diff <= 0) {
    return s;
  }

  return fillCharacter.repeat(diff) + s;
}
