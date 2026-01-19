/** Parameters for weight mapping object. */
export interface WeightMapperParams {
  exact: Record<string, string>;
  prefix: Record<string, string>;
  suffix: Record<string, string>;
  substring: Record<string, string>;
  autoCamelCase: boolean;
}

/**
 * Converts model weights in one format to another format for use in JavaScript,
 * based on substring matchers.
 */
export class WeightMapper {
  readonly #params: WeightMapperParams;

  constructor(params: Partial<WeightMapperParams>) {
    this.#params = {
      exact: params.exact ?? {},
      prefix: params.prefix ?? {},
      suffix: params.suffix ?? {},
      substring: params.substring ?? {},
      autoCamelCase: params.autoCamelCase ?? false,
    };
  }

  mapKey(key: string): string {
    let mappedKey = key;
    for (const [from, to] of Object.entries(this.#params.exact)) {
      if (mappedKey === from) {
        mappedKey = to;
      }
    }
    for (const [from, to] of Object.entries(this.#params.prefix)) {
      if (mappedKey.startsWith(from)) {
        mappedKey = to + mappedKey.slice(from.length);
      }
    }
    for (const [from, to] of Object.entries(this.#params.suffix)) {
      if (mappedKey.endsWith(from)) {
        mappedKey = mappedKey.slice(0, -from.length) + to;
      }
    }
    for (const [from, to] of Object.entries(this.#params.substring)) {
      mappedKey = mappedKey.replaceAll(from, to);
    }
    if (this.#params.autoCamelCase) {
      mappedKey = mappedKey.replace(/_([a-z])/g, (_, char) =>
        char.toUpperCase(),
      );
    }
    return mappedKey;
  }

  unmapKey(key: string): string {
    let unmappedKey = key;
    // Apply the operations in reverse order.
    if (this.#params.autoCamelCase) {
      unmappedKey = unmappedKey.replace(
        /[A-Z]/g,
        (char) => `_${char.toLowerCase()}`,
      );
    }
    for (const [from, to] of Object.entries(this.#params.substring).reverse()) {
      unmappedKey = unmappedKey.replaceAll(to, from);
    }
    for (const [from, to] of Object.entries(this.#params.suffix).reverse()) {
      if (unmappedKey.endsWith(to)) {
        unmappedKey = unmappedKey.slice(0, -to.length) + from;
      }
    }
    for (const [from, to] of Object.entries(this.#params.prefix).reverse()) {
      if (unmappedKey.startsWith(to)) {
        unmappedKey = from + unmappedKey.slice(to.length);
      }
    }
    for (const [from, to] of Object.entries(this.#params.exact).reverse()) {
      if (unmappedKey === to) {
        unmappedKey = from;
      }
    }
    return unmappedKey;
  }

  mapObject<T>(obj: Record<string, T>): Record<string, T> {
    return Object.fromEntries(
      Object.entries(obj).map(([key, value]) => [this.mapKey(key), value]),
    );
  }

  unmapObject<T>(obj: Record<string, T>): Record<string, T> {
    return Object.fromEntries(
      Object.entries(obj).map(([key, value]) => [this.unmapKey(key), value]),
    );
  }
}
