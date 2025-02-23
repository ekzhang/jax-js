/** General class for pretty-printing expressions with indentation. */
export class PPrint {
  constructor(
    readonly indents: number[],
    readonly lines: string[],
  ) {}

  /** Add a fixed amount of indentation to each line. */
  indent(spaces: number): PPrint {
    return new PPrint(
      this.indents.map((i) => i + spaces),
      this.lines,
    );
  }

  /** Concatenate two or more pretty-printed expressions. */
  concat(...items: PPrint[]): PPrint {
    return new PPrint(
      (this.indents ?? []).concat(...items.map((i) => i.indents)),
      (this.lines ?? []).concat(...items.map((i) => i.lines)),
    );
  }

  /** Stack one block to the right of another one, sharing 1 common line. */
  stack(other: PPrint): PPrint {
    if (!other.lines.length) return this;
    if (!this.lines.length) return other;
    const indent = this.indents[this.indents.length - 1];
    const s = this.lines[this.lines.length - 1];
    const indentedBlock = other.indent(indent + s.length);
    return new PPrint(
      this.indents.concat(indentedBlock.indents.slice(1)),
      this.lines
        .slice(0, -1)
        .concat(
          s + " ".repeat(other.indents[0]) + other.lines[0],
          ...indentedBlock.lines.slice(1),
        ),
    );
  }

  /** Combine this block of lines into a formatted string. */
  toString(): string {
    return this.lines
      .map((line, i) => " ".repeat(this.indents[i]) + line)
      .join("\n");
  }

  static pp(s: Stringable): PPrint {
    const lines = s.toString().split("\n");
    return new PPrint(Array(lines.length).fill(0), lines);
  }
}

interface Stringable {
  toString(): string;
}
