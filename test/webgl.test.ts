import { defaultDevice, init, numpy as np } from "@jax-js/jax";
import { beforeEach, expect, suite, test } from "vitest";

const devicesAvailable = await init("webgl");

suite("WebGL materialized comparisons", () => {
  beforeEach(({ skip }) => {
    if (!devicesAvailable.includes("webgl")) skip();
    defaultDevice("webgl");
  });

  // At least 128 elements are needed to avoid the small-expression CPU path.
  // The odd length also exercises a partially filled texture.
  const values = Array.from({ length: 131 }, (_, i) => (i % 7) - 3);

  test.each(["less", "greater", "lessEqual"] as const)(
    "%s survives readback and reuse as a mask",
    async (op) => {
      const expected = values.map((x) =>
        op === "less" ? x < 0 : op === "greater" ? x > 0 : x <= 0,
      );
      for (const arrayRhs of [false, true]) {
        const x = np.array(values);
        const rhs = arrayRhs ? np.zeros([values.length]) : 0;
        const mask = x[op](rhs);

        // Force the comparison to be stored before it is used by another op.
        expect(mask.dtype).toBe(np.bool);
        expect(mask.ref.js()).toEqual(expected);
        expect(await mask.ref.jsAsync()).toEqual(expected);
        expect(Array.from(await mask.ref.data())).toEqual(
          expected.map(Number),
        );
        const selected = np.where(mask, -1, 1);
        expect(await selected.jsAsync()).toEqual(
          expected.map((value) => (value ? -1 : 1)),
        );
      }
    },
  );

  test.each([false, true])("materializes a full %s mask", async (value) => {
    const mask = np.full([values.length], value, { dtype: np.bool });
    const expected = new Int32Array(values.length).fill(Number(value));
    expect(mask.ref.dataSync()).toEqual(expected);
    expect(await mask.data()).toEqual(expected);
  });
});
