import { Matrix } from "../matrix";

const DECIMALS = 3;

test("Matrix.random", () => {
  const a = Matrix.zeros(6, 1);
  expect([a.rows, a.columns]).toEqual([6, 1]);
  expect(
    a.rowMajorOrderEntries().every((entry) => entry >= 0 && entry < 1)
  ).toBe(true);

  const b = Matrix.zeros(2, 3);
  expect([b.rows, b.columns]).toEqual([2, 3]);
  expect(
    b.rowMajorOrderEntries().every((entry) => entry >= 0 && entry < 1)
  ).toBe(true);
});

test("Matrix.zeros", () => {
  expect(Matrix.zeros(6, 1).print(DECIMALS)).toMatchSnapshot();
  expect(Matrix.zeros(2, 3).print(DECIMALS)).toMatchSnapshot();
});

test("Matrix.fromRows", () => {
  const a = Matrix.fromRows([
    [1, 2],
    [3, 4],
    [5, 6],
  ]);
  expect(a.print(DECIMALS)).toMatchSnapshot();

  const b = Matrix.fromRows([[7], [8], [9]]);
  expect(b.print(DECIMALS)).toMatchSnapshot();

  const c = Matrix.fromRows([
    [1, -2, 3, 4],
    [-5, -6, 7, -8],
  ]);
  expect(c.print(DECIMALS)).toMatchSnapshot();
});

test("Matrix.fromRows rejects jagged arrays", () => {
  expect(() => {
    Matrix.fromRows([[1, 2], [3], [4, 5]]);
  }).toThrow();
});

test("Matrix.columnVector", () => {
  const a = Matrix.columnVector([1, -2, 3.5]);
  expect(a.rows).toBe(3);
  expect(a.columns).toBe(1);
  expect(a.rowMajorOrderEntries()).toEqual([1, -2, 3.5]);
});

test("Matrix.prototype.clone", () => {
  const a = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);
  const clone = a.clone();

  expectEquals(clone, a);

  // Clones should have the same entries as the original,
  // but they should not reference the same array as the original.
  expect(clone.rowMajorOrderEntries()).toEqual(a.rowMajorOrderEntries());
  expect(clone.rowMajorOrderEntries()).not.toBe(a.rowMajorOrderEntries());
});

test("Matrix.prototype.mutMultiplyScalar", () => {
  const a = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);
  a.mutMultiplyScalar(42);

  expectEquals(
    a,
    Matrix.fromRows([
      [-1 * 42, 2 * 42],
      [3 * 42, -4 * 42],
      [-5 * 42, -6 * 42],
    ])
  );

  const b = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);
  b.mutMultiplyScalar(-0.3);

  expectEquals(
    b,
    Matrix.fromRows([
      [-1 * -0.3, 2 * -0.3],
      [3 * -0.3, -4 * -0.3],
      [-5 * -0.3, -6 * -0.3],
    ])
  );
});

test("Matrix.prototype.mutAdd", () => {
  const a = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);
  a.mutAdd(
    Matrix.fromRows([
      [7.3, -8],
      [-9, 10.5],
      [-11, 12],
    ])
  );

  expectEquals(
    a,
    Matrix.fromRows([
      [-1 + 7.3, 2 + -8],
      [3 + -9, -4 + 10.5],
      [-5 + -11, -6 + 12],
    ])
  );
});

test("Matrix.prototye.mutAdd throws if RHS matrix has different dimensions", () => {
  const a = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);
  const b = Matrix.fromRows([
    [-1, 2, 3],
    [-4, 5, 6],
  ]);

  expect(() => {
    a.mutAdd(b);
  }).toThrow();
});

test("Matrix.prototype.immutMultiply", () => {
  const a = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);
  const b = Matrix.fromRows([
    [-7, 8, 9],
    [-10, 11, 12],
  ]);

  expectEquals(
    a.immutMultiply(b),
    Matrix.fromRows([
      [-13, 14, 15],
      [19, -20, -21],
      [95, -106, -117],
    ])
  );

  expectEquals(
    b.immutMultiply(a),
    Matrix.fromRows([
      [-14, -100],
      [-17, -136],
    ])
  );
});

test("Matrix.prototype.immutMultiply throws if this.columns !== RHS.rows", () => {
  const a = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);

  expect(() => {
    a.immutMultiply(a);
  }).toThrow();

  const b = Matrix.fromRows([
    [-7, 8, 9],
    [-10, 11, 12],
    [-13, 14, 15],
  ]);

  expect(() => {
    a.immutMultiply(b);
  }).toThrow();
});

test("Matrix.prototype.mutHadamard", () => {
  const a = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);
  a.mutHadamard(
    Matrix.fromRows([
      [-7.1, 8],
      [9, -10.2],
      [11.9, 12],
    ])
  );

  expectEquals(
    a,
    Matrix.fromRows([
      [-1 * -7.1, 2 * 8],
      [3 * 9, -4 * -10.2],
      [-5 * 11.9, -6 * 12],
    ])
  );
});

test("Matrix.prototype.mutHadamard throws if RHS matrix has different dimensions", () => {
  const a = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);
  const b = Matrix.fromRows([
    [-7, 8, 9],
    [-10, 11, 12],
  ]);

  expect(() => {
    a.mutHadamard(b);
  }).toThrow();

  expect(() => {
    b.mutHadamard(a);
  }).toThrow();
});

test("Matrix.prototype.immutTranspose", () => {
  const a = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);
  const aTrans = a.immutTranspose();
  expectEquals(
    aTrans,
    Matrix.fromRows([
      [-1, 3, -5],
      [2, -4, -6],
    ])
  );
});

test("Matrix.prototype.rowMajorOrderEntries", () => {
  const a = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);
  expect(a.rowMajorOrderEntries()).toEqual([-1, 2, 3, -4, -5, -6]);
});

test("Matrix.prototype.immutApplyElementwise", () => {
  function cube(x: number): number {
    return Math.pow(x, 3);
  }

  const a = Matrix.fromRows([
    [-1, 2],
    [3, -4],
    [-5, -6],
  ]);
  expectEquals(
    a.immutApplyElementwise(cube),
    Matrix.fromRows([
      [cube(-1), cube(2)],
      [cube(3), cube(-4)],
      [cube(-5), cube(-6)],
    ])
  );
});

function expectEquals(a: Matrix, b: Matrix): void {
  expect(a.print(DECIMALS)).toBe(b.print(DECIMALS));
}
