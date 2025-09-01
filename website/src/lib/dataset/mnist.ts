import { cachedFetch } from "@jax-js/opfs";

// CORS-enabled version of https://github.com/cvdfoundation/mnist
const mnistLinks = {
  train: {
    images: "https://cdn.jsdelivr.net/gh/fgnt/mnist/train-images-idx3-ubyte.gz",
    labels: "https://cdn.jsdelivr.net/gh/fgnt/mnist/train-labels-idx1-ubyte.gz",
  },
  test: {
    images: "https://cdn.jsdelivr.net/gh/fgnt/mnist/t10k-images-idx3-ubyte.gz",
    labels: "https://cdn.jsdelivr.net/gh/fgnt/mnist/t10k-labels-idx1-ubyte.gz",
  },
};

// "The IDX file format is a simple format for vectors and multidimensional
// matrices of various numerical types. The basic format is
//
// magic number
// size in dimension 0
// size in dimension 1
// size in dimension 2
// .....
// size in dimension N
// data
//
// The magic number is an integer (MSB first). The first 2 bytes are always 0.
// The third byte codes the type of the data:
//
// 0x08: unsigned byte
// 0x09: signed byte
// 0x0B: short (2 bytes)
// 0x0C: int (4 bytes)
// 0x0D: float (4 bytes)
// 0x0E: double (8 bytes)
async function fetchIdxFile(
  url: string,
): Promise<{ shape: number[]; data: Int32Array | Float32Array }> {
  const bytes = await cachedFetch(url);
  const stream = new Blob([bytes])
    .stream()
    .pipeThrough(new DecompressionStream("gzip"));
  const buffer = await new Response(stream).arrayBuffer();
  let view = new DataView(buffer);

  const dataType = view.getUint8(2);
  const rank = view.getUint8(3);

  const shape: number[] = [];
  for (let i = 0; i < rank; i++) {
    shape.push(view.getUint32(4 + i * 4, false)); // Big-endian
  }
  const size = shape.reduce((a, b) => a * b, 1);

  // Now, read the data.
  view = new DataView(buffer, 4 + rank * 4);
  let data: Int32Array | Float32Array;

  switch (dataType) {
    case 0x08: // unsigned byte
      data = new Int32Array(size);
      for (let i = 0; i < size; i++) data[i] = view.getUint8(i);
      break;

    case 0x09: // signed byte
      data = new Int32Array(size);
      for (let i = 0; i < size; i++) data[i] = view.getInt8(i);
      break;

    case 0x0b: // short (2 bytes)
      data = new Int32Array(size);
      for (let i = 0; i < size; i++) data[i] = view.getInt16(i * 2, false); // Big-endian
      break;

    case 0x0c: // int (4 bytes)
      data = new Int32Array(size);
      for (let i = 0; i < size; i++) data[i] = view.getInt32(i * 4, false); // Big-endian
      break;

    case 0x0d: // float (4 bytes)
      data = new Float32Array(size);
      for (let i = 0; i < size; i++) data[i] = view.getFloat32(i * 4, false); // Big-endian
      break;

    case 0x0e: // double (8 bytes)
      data = new Float32Array(size);
      for (let i = 0; i < size; i++) data[i] = view.getFloat64(i * 8, false); // Big-endian
      break;

    default:
      throw new Error(`Unsupported data type: ${dataType}`);
  }

  return { shape, data };
}

export async function fetchMnist() {
  const results = await Promise.all([
    fetchIdxFile(mnistLinks.train.images),
    fetchIdxFile(mnistLinks.train.labels),
    fetchIdxFile(mnistLinks.test.images),
    fetchIdxFile(mnistLinks.test.labels),
  ]);

  return {
    train: {
      images: results[0],
      labels: results[1],
    },
    test: {
      images: results[2],
      labels: results[3],
    },
  };
}
