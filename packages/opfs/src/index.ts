/** FileInfo object containing metadata about cached files. */
export interface FileInfo {
  name: string;
  lastModified: Date;
  size: number;
}

function isOPFSSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    "storage" in navigator &&
    "getDirectory" in navigator.storage
  );
}

function fileToInfo(name: string, file: File): FileInfo {
  return {
    name,
    lastModified: new Date(file.lastModified),
    size: file.size,
  };
}

/** Escape problematic characters in keys for safe file system usage. */
function escapeKey(name: string): string {
  // Use hex encoding (case-insensitive, filesystem-safe)
  const encoder = new TextEncoder();
  const bytes = encoder.encode(name);
  const hex = Array.from(bytes)
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
  return `blob-${hex}`;
}

function unescapeKey(key: string): string {
  try {
    if (!key.startsWith("blob-")) return key;
    const hex = key.slice(5); // Remove "blob-" prefix
    const bytes = new Uint8Array(
      hex.match(/.{2}/g)?.map((h) => parseInt(h, 16)) ?? [],
    );
    return new TextDecoder().decode(bytes);
  } catch {
    return key;
  }
}

export class OPFS {
  #root: FileSystemDirectoryHandle | null = null;

  async #getRoot(): Promise<FileSystemDirectoryHandle> {
    if (!this.#root) {
      if (!isOPFSSupported()) {
        throw new Error("OPFS is not supported in this environment");
      }
      const dir = await navigator.storage.getDirectory();
      this.#root = await dir.getDirectoryHandle("jax-js", { create: true });
    }
    return this.#root;
  }

  /** Write data to OPFS with the given key. */
  async write(name: string, data: Uint8Array): Promise<void> {
    const root = await this.#getRoot();
    const key = escapeKey(name);
    const fileHandle = await root.getFileHandle(key, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(data);
    await writable.close();
  }

  async #getFile(name: string): Promise<File | null> {
    const root = await this.#getRoot();
    const key = escapeKey(name);
    try {
      const fileHandle = await root.getFileHandle(key);
      return await fileHandle.getFile();
    } catch (error) {
      if (error instanceof DOMException && error.name === "NotFoundError") {
        return null;
      }
      throw error;
    }
  }

  /** Read data from OPFS with the given key. */
  async read(name: string): Promise<Uint8Array | null> {
    const file = await this.#getFile(name);
    if (!file) return null;
    return new Uint8Array(await file.arrayBuffer());
  }

  /** Get file information for the given key. */
  async info(name: string): Promise<FileInfo | null> {
    const file = await this.#getFile(name);
    if (!file) return null;
    return fileToInfo(name, file);
  }

  /** List all files in OPFS. */
  async list(): Promise<FileInfo[]> {
    const root = await this.#getRoot();
    const files: FileInfo[] = [];

    for await (const [key, handle] of root.entries()) {
      if (handle.kind === "file") {
        const name = unescapeKey(key);
        try {
          const file = await (handle as FileSystemFileHandle).getFile();
          files.push(fileToInfo(name, file));
        } catch (error) {
          if (error instanceof DOMException && error.name === "NotFoundError") {
            continue; // File was deleted
          }
          throw error;
        }
      }
    }

    return files;
  }

  /** Remove a file from OPFS and return its info if it existed. */
  async remove(name: string): Promise<FileInfo | null> {
    const root = await this.#getRoot();
    const file = await this.#getFile(name);
    if (!file) return null;
    const info = fileToInfo(name, file);
    const key = escapeKey(name);
    try {
      await root.removeEntry(key);
    } catch (error) {
      if (error instanceof DOMException && error.name === "NotFoundError") {
        return null; // Already deleted, race condition
      }
      throw error;
    }
    return info;
  }

  /** Clear all files from OPFS. */
  async clear(): Promise<void> {
    const root = await this.#getRoot();
    for await (const key of root.keys()) {
      try {
        await root.removeEntry(key);
      } catch (error) {
        if (error instanceof DOMException && error.name === "NotFoundError") {
          continue; // Already deleted
        }
        throw error;
      }
    }
  }
}

/** Primary OPFS instance for blob caching. */
export const opfs = new OPFS();

/** Cached fetch that stores responses in OPFS and returns the data as `Uint8Array`. */
export async function cachedFetch(
  input: string | URL,
  init?: RequestInit,
): Promise<Uint8Array> {
  const url = typeof input === "string" ? input : String(input);

  const cachedData = await opfs.read(url);
  if (cachedData !== null) {
    return cachedData;
  }

  const resp = await fetch(input, init);
  if (!resp.ok) {
    throw new Error(
      `Failed to fetch ${url}: ${resp.status} ${resp.statusText}`,
    );
  }
  const data = await resp.bytes();
  try {
    await opfs.write(url, data);
  } catch (cacheError) {
    console.warn(`Failed to cache response in OPFS for ${url}:`, cacheError);
  }

  return data;
}
