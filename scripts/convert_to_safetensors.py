# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "rich",
#     "safetensors",
#     "typer",
#     "torch",
#     "packaging",
#     "numpy",
# ]
# ///

import typer
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Annotated
from safetensors.torch import save_file, load_file
from rich.console import Console
from rich.tree import Tree
from rich.text import Text
from rich.table import Table


class CompactTree(Tree):
    """Tree with reduced indentation spacing"""

    ASCII_GUIDES = ("  ", "| ", "+ ", "` ")
    TREE_GUIDES = [
        ("  ", "│ ", "├ ", "└ "),
        ("  ", "┃ ", "┣ ", "┗ "),
        ("  ", "║ ", "╠ ", "╚ "),
    ]


console = Console()
app = typer.Typer(
    help="Convert PyTorch or safetensors files to safetensors format",
    add_completion=False,
)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def get_tensor_size(tensor) -> int:
    """Calculate tensor size in bytes"""
    return tensor.numel() * tensor.element_size()


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            if unit == "B":
                return f"{int(size_bytes)} B"
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def load_tensors(file_path: Path) -> Dict[str, torch.Tensor]:
    """Load tensors from either .pt or .safetensors file"""
    if file_path.suffix.lower() == ".pt":
        data = torch.load(file_path, map_location="cpu")
        if isinstance(data, dict):
            return data
        else:
            return {"tensor": data}
    elif file_path.suffix.lower() == ".safetensors":
        return load_file(str(file_path))
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def convert_tensor_dtype(
    tensor: torch.Tensor, target_dtype: torch.dtype
) -> torch.Tensor:
    """Convert tensor to target dtype"""
    if tensor.dtype == target_dtype:
        return tensor
    return tensor.to(target_dtype)


def parse_conversion_rules(
    convert_options: list[str],
) -> Dict[torch.dtype, torch.dtype]:
    """Parse conversion rules like 'float32:float16' into a mapping"""
    rules = {}
    for option in convert_options:
        if ":" not in option:
            raise ValueError(
                f"Invalid conversion format: {option}. Use 'from_dtype:to_dtype'"
            )

        from_dtype_str, to_dtype_str = option.split(":", 1)

        if from_dtype_str not in DTYPE_MAP:
            raise ValueError(f"Unknown source dtype: {from_dtype_str}")
        if to_dtype_str not in DTYPE_MAP:
            raise ValueError(f"Unknown target dtype: {to_dtype_str}")

        from_dtype = DTYPE_MAP[from_dtype_str]
        to_dtype = DTYPE_MAP[to_dtype_str]
        rules[from_dtype] = to_dtype

    return rules


def apply_conversion_rules(
    tensors: Dict[str, torch.Tensor], rules: Dict[torch.dtype, torch.dtype]
) -> Dict[str, torch.Tensor]:
    """Apply conversion rules to tensors"""
    converted_tensors = {}

    for name, tensor in tensors.items():
        if tensor.dtype in rules:
            converted_tensors[name] = convert_tensor_dtype(tensor, rules[tensor.dtype])
        else:
            converted_tensors[name] = tensor

    return converted_tensors


def get_dtype_summary(tensors: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, Any]]:
    """Get summary statistics by dtype"""
    summary = {}

    for name, tensor in tensors.items():
        dtype_str = str(tensor.dtype).replace("torch.", "")
        size = get_tensor_size(tensor)
        num_elements = tensor.numel()

        if dtype_str not in summary:
            summary[dtype_str] = {"count": 0, "total_size": 0, "total_elements": 0}

        summary[dtype_str]["count"] += 1
        summary[dtype_str]["total_size"] += size
        summary[dtype_str]["total_elements"] += num_elements

    return summary


def display_dtype_summary(
    before_tensors: Dict[str, torch.Tensor], after_tensors: Dict[str, torch.Tensor]
):
    """Display a comparison table of dtypes before and after conversion"""
    before_summary = get_dtype_summary(before_tensors)
    after_summary = get_dtype_summary(after_tensors)

    # Get all unique dtypes from both before and after
    all_dtypes = set(before_summary.keys()) | set(after_summary.keys())

    table = Table(title="Data Type Summary")
    table.add_column("Data Type", style="bold blue")
    table.add_column("Before", justify="right")
    table.add_column("Before Size", justify="right", style="yellow")
    table.add_column("After", justify="right")
    table.add_column("After Size", justify="right", style="yellow")
    table.add_column("Size Change", justify="right")

    for dtype in sorted(all_dtypes):
        before_data = before_summary.get(
            dtype, {"count": 0, "total_size": 0, "total_elements": 0}
        )
        after_data = after_summary.get(
            dtype, {"count": 0, "total_size": 0, "total_elements": 0}
        )

        before_count = before_data["count"]
        before_size = before_data["total_size"]
        before_elements = before_data["total_elements"]
        after_count = after_data["count"]
        after_size = after_data["total_size"]
        after_elements = after_data["total_elements"]

        # Calculate size change with color coding
        if before_size == 0 and after_size == 0:
            change = "—"
        elif before_size == 0:
            change = f"[red]+{format_size(after_size)}[/red]"
        elif after_size == 0:
            change = f"[green]-{format_size(before_size)}[/green]"
        else:
            size_diff = after_size - before_size
            if size_diff > 0:
                change = f"[red]+{format_size(size_diff)}[/red]"
            elif size_diff < 0:
                change = f"[green]-{format_size(abs(size_diff))}[/green]"
            else:
                change = "—"

        # Format element counts with tensor counts in parentheses
        before_str = (
            f"{before_elements:,} ({before_count} T)" if before_count > 0 else "—"
        )
        after_str = f"{after_elements:,} ({after_count} T)" if after_count > 0 else "—"

        # Only show rows that have data in before or after
        if before_count > 0 or after_count > 0:
            table.add_row(
                dtype,
                before_str,
                format_size(before_size) if before_size > 0 else "—",
                after_str,
                format_size(after_size) if after_size > 0 else "—",
                change,
            )

    console.print()
    console.print(table)


def build_tree_structure(tensors: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Build hierarchical structure from flat tensor names"""
    tree = {}

    for name, tensor in tensors.items():
        parts = name.split(".")
        current = tree

        for i, part in enumerate(parts[:-1]):
            if part not in current:
                current[part] = {"_children": {}, "_size": 0}
            current = current[part]["_children"]

        final_part = parts[-1]
        tensor_size = get_tensor_size(tensor)
        current[final_part] = {
            "_tensor": tensor,
            "_size": tensor_size,
            "_shape": list(tensor.shape),
            "_dtype": str(tensor.dtype).replace("torch.", ""),
        }

    def calculate_sizes(node):
        if "_tensor" in node:
            return node["_size"]

        total_size = 0
        for key, child in node.get("_children", {}).items():
            child_size = calculate_sizes(child)
            total_size += child_size

        node["_size"] = total_size
        return total_size

    calculate_sizes({"_children": tree})
    return tree


def add_tree_nodes(tree: CompactTree, structure: Dict[str, Any], parent_name: str = ""):
    """Recursively add nodes to rich tree"""
    for name, data in structure.items():
        if name.startswith("_"):
            continue

        full_name = f"{parent_name}.{name}" if parent_name else name
        size_bytes = data["_size"]
        size_str = format_size(size_bytes) if size_bytes >= 1024 * 1024 else ""

        if "_tensor" in data:
            shape_str = "x".join(map(str, data["_shape"]))
            dtype_str = data["_dtype"]
            label = Text(f"{name} ", style="bold blue")
            label.append(f"[{shape_str}] ", style="dim")
            label.append(f"{dtype_str} ", style="green")
            if size_str:
                label.append(f"({size_str})", style="yellow")
            tree.add(label)
        else:
            label = Text(f"{name}/ ", style="bold magenta")
            if size_str:
                label.append(f"({size_str})", style="yellow")
            branch = tree.add(label)
            add_tree_nodes(branch, data.get("_children", {}), full_name)


def display_tensor_tree(
    tensors: Dict[str, torch.Tensor], title: str = "Tensor Structure"
):
    """Display tensors as a tree with hierarchy and sizes"""
    structure = build_tree_structure(tensors)

    total_size = sum(get_tensor_size(tensor) for tensor in tensors.values())
    tree = CompactTree(
        f"[bold cyan]{title}[/bold cyan] [yellow]({format_size(total_size)})[/yellow]",
        guide_style="dim white",
    )

    add_tree_nodes(tree, structure)
    console.print(tree)


@app.command()
def convert(
    input_file: Annotated[Path, typer.Argument(help="Input .pt or .safetensors file")],
    output_file: Annotated[
        Optional[Path],
        typer.Option(
            "--output",
            "-o",
            help="Output .safetensors file (default: input_file.safetensors)",
        ),
    ] = None,
    convert: Annotated[
        list[str],
        typer.Option(
            "--convert",
            "-c",
            help="Selective dtype conversion rules (e.g., 'float32:float16', 'int64:int32'). Can be specified multiple times.",
        ),
    ] = [],
    show_input: Annotated[
        bool,
        typer.Option("--show-input/--no-show-input", help="Show input file structure"),
    ] = True,
    show_output: Annotated[
        bool,
        typer.Option(
            "--show-output/--no-show-output", help="Show output file structure"
        ),
    ] = True,
):
    """Convert PyTorch or safetensors file to safetensors format"""

    if not input_file.exists():
        console.print(f"[red]Error: Input file {input_file} does not exist[/red]")
        raise typer.Exit(1)

    if output_file is None:
        output_file = input_file.with_suffix(".safetensors")

    try:
        console.print(f"[blue]Loading tensors from {input_file}...[/blue]")
        original_tensors = load_tensors(input_file)

        if show_input:
            console.print()
            display_tensor_tree(original_tensors, f"Input: {input_file.name}")

        # Keep original for comparison
        tensors = original_tensors

        if convert:
            console.print(
                f"\n[blue]Applying selective conversions: {', '.join(convert)}...[/blue]"
            )
            conversion_rules = parse_conversion_rules(convert)
            tensors = apply_conversion_rules(tensors, conversion_rules)

        console.print(f"\n[blue]Saving to {output_file}...[/blue]")
        save_file(tensors, str(output_file))

        if show_output:
            console.print()
            display_tensor_tree(tensors, f"Output: {output_file.name}")

        console.print(f"\n[green]✓ Successfully converted to {output_file}[/green]")

        display_dtype_summary(original_tensors, tensors)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
