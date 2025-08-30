/** Simple tensor memory allocator for WebAssembly linear memory. */
export class WasmAllocator {
  #memory: WebAssembly.Memory;
  #headPtr: number;
  #freeLists: Map<number, number[]>;
  #allocatedBuffers: Map<number, number>; // ptr -> sizeClass

  constructor(memory: WebAssembly.Memory) {
    this.#memory = memory;
    this.#headPtr = 64; // Address 0 is reserved for empty slices.
    this.#freeLists = new Map();
    this.#allocatedBuffers = new Map();
  }

  malloc(size: number): number {
    if (size === 0) return 0;

    const sizeClass = this.#findSizeClass(size);
    const freeList = this.#freeLists.get(sizeClass);

    let ptr: number;
    if (freeList && freeList.length > 0) {
      ptr = freeList.pop()!;
    } else {
      ptr = this.#bumpAlloc(sizeClass);
    }

    this.#allocatedBuffers.set(ptr, sizeClass);
    return ptr;
  }

  free(ptr: number): void {
    if (ptr === 0) return;

    const sizeClass = this.#allocatedBuffers.get(ptr);
    if (sizeClass === undefined) {
      throw new Error(`Attempting to free unallocated pointer: ${ptr}`);
    }

    const freeList = this.#freeLists.get(sizeClass);
    if (freeList) freeList.push(ptr);
    else this.#freeLists.set(sizeClass, [ptr]);
    this.#allocatedBuffers.delete(ptr);
  }

  #bumpAlloc(size: number): number {
    const ptr = this.#headPtr;
    size = (size + 63) & -64; // Align to 64 bytes, like Arrow.
    this.#headPtr += size;
    if (ptr + size > this.#memory.buffer.byteLength) {
      // Note: 4 GiB = max memory32 size
      // https://spidermonkey.dev/blog/2025/01/15/is-memory64-actually-worth-using.html
      this.#memory.grow(
        ((ptr + size + 65535) >> 16) - (this.#memory.buffer.byteLength >> 16),
      );
    }
    return ptr;
  }

  #findSizeClass(size: number): number {
    // Small sizes: 64-byte increments from 64 to 512.
    if (size <= 512) {
      return (size + 63) & -64;
    }
    // Medium sizes: 768 (512+256), then 256-byte increments from 1024 to 2048.
    if (size <= 2048) {
      return (size + 511) & -512;
    }
    // Large sizes: powers of 2 from 4 KiB to 64 KiB.
    if (size <= 65536) {
      let sizeClass = 4096;
      while (sizeClass < size) sizeClass *= 2;
      return sizeClass;
    }
    // Very large sizes: 64 KiB increments starting from 128 KiB.
    return (size + 65535) & -65536;
  }

  // Debug methods
  getStats(): { totalAllocated: number; freeListSizes: Map<number, number> } {
    const freeListSizes = new Map<number, number>();
    for (const [sizeClass, freeList] of this.#freeLists) {
      if (freeList.length > 0) {
        freeListSizes.set(sizeClass, freeList.length);
      }
    }

    return {
      totalAllocated: this.#headPtr,
      freeListSizes,
    };
  }
}
