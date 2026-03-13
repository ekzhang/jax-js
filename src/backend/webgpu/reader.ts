/**
 * Graphics state used to synchronously read data from WebGPU buffers.
 *
 * This trick is borrowed from TensorFlow.js. Basically, the idea is to create
 * an offscreen canvas with one pixel for every 4 bytes ("device storage"), then
 * configure it with a WebGPU context. Copy the buffer to a texture, then draw
 * the canvas onto another offscreen canvas with '2d' context ("host storage").
 *
 * Once it's on host storage, we can use `getImageData()` to read the pixels
 * from the image directly.
 *
 * We use 256x256 canvases here (256 KiB). The performance of this is bad
 * because it involves multiple data copies, but it still works. We also
 * actually need to copy the image twice: once in "opaque" mode for the RGB
 * values, and once in "premultiplied" mode for the alpha channel.
 *
 * https://github.com/tensorflow/tfjs/blob/tfjs-v4.22.0/tfjs-backend-webgpu/src/backend_webgpu.ts#L379
 */
export class SyncReader {
  static readonly alphaModes: GPUCanvasAlphaMode[] = [
    "opaque",
    "premultiplied",
  ];
  static readonly width = 256;
  static readonly height = 256;

  initialized = false;
  deviceStorage?: OffscreenCanvas[];
  deviceContexts?: GPUCanvasContext[];
  hostStorage?: OffscreenCanvas;
  hostContext?: OffscreenCanvasRenderingContext2D;

  constructor(readonly device: GPUDevice) {}

  #init() {
    // Some platforms, such as Deno do not support `OffscreenCanvas`. Add an
    // error message to make this clear.
    if (typeof OffscreenCanvas === "undefined") {
      throw new Error(
        "OffscreenCanvas is not available in this environment, so you cannot " +
          "read data from WebGPU synchronously. Consider using the async API.",
      );
    }

    const makeCanvas = () =>
      new OffscreenCanvas(SyncReader.width, SyncReader.height);
    this.deviceStorage = SyncReader.alphaModes.map(makeCanvas);
    this.deviceContexts = this.deviceStorage.map((canvas, i) => {
      const context = canvas.getContext("webgpu")!;
      context.configure({
        device: this.device,
        // rgba8unorm is not supported on Chrome for macOS.
        // https://bugs.chromium.org/p/chromium/issues/detail?id=1298618
        format: "bgra8unorm",
        usage: GPUTextureUsage.COPY_DST,
        alphaMode: SyncReader.alphaModes[i],
      });
      return context;
    });
    this.hostStorage = makeCanvas();
    this.hostContext = this.hostStorage.getContext("2d", {
      willReadFrequently: true,
    })!;
    this.initialized = true;
  }

  read(
    buffer: GPUBuffer,
    start: number,
    count: number,
  ): Uint8Array<ArrayBuffer> {
    if (!this.initialized) this.#init();

    const deviceStorage = this.deviceStorage!;
    const deviceContexts = this.deviceContexts!;
    const hostContext = this.hostContext!;

    // WebGPU has a fundamental alignment requirement of 4 bytes when copying or
    // accessing buffers, so we round up the size here.
    const pixelsSize = Math.ceil(count / 4);
    const bytesPerRow = SyncReader.width * 4;
    const valsGPU = new ArrayBuffer(pixelsSize * 4);

    for (let i = 0; i < deviceContexts.length; i++) {
      const texture = deviceContexts[i].getCurrentTexture();
      // Read data using a (width, height) image at `offset` in valsGPU.
      const readData = (width: number, height: number, offset: number) => {
        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToTexture(
          { buffer, bytesPerRow, offset: offset + start },
          { texture },
          { width, height, depthOrArrayLayers: 1 },
        );
        const commandBuffer = encoder.finish();
        this.device.queue.submit([commandBuffer]);

        hostContext.clearRect(0, 0, width, height);
        hostContext.drawImage(deviceStorage[i], 0, 0);
        const values = hostContext.getImageData(0, 0, width, height).data;
        const span = new Uint8ClampedArray(valsGPU, offset, 4 * width * height);
        const alphaMode = SyncReader.alphaModes[i];
        for (let k = 0; k < span.length; k += 4) {
          if (alphaMode === "premultiplied") {
            span[k + 3] = values[k + 3];
          } else {
            span[k] = values[k + 2]; // opaque (BGRA)
            span[k + 1] = values[k + 1];
            span[k + 2] = values[k];
          }
        }
      };

      const pixelsPerCanvas = SyncReader.width * SyncReader.height;
      const wholeChunks = Math.floor(pixelsSize / pixelsPerCanvas);
      let remainder = pixelsSize % pixelsPerCanvas;
      const remainderRows = Math.floor(remainder / SyncReader.width);
      remainder = remainder % SyncReader.width;

      let offset = 0;
      // Read entire canvases.
      for (let j = 0; j < wholeChunks; j++) {
        readData(SyncReader.width, SyncReader.height, offset);
        offset += pixelsPerCanvas * 4;
      }
      // Read a partial canvas with whole rows.
      if (remainderRows > 0) {
        readData(SyncReader.width, remainderRows, offset);
        offset += remainderRows * SyncReader.width * 4;
      }
      // Read a partial canvas with some columns in the first row.
      if (remainder > 0) readData(remainder, 1, offset);
    }

    return new Uint8Array(valsGPU, 0, count);
  }
}
