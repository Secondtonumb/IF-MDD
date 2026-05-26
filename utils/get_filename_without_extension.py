from pathlib import Path


def get_filename_without_extension(path):
    """Return the annotation filename stem for result naming."""
    if isinstance(path, (list, tuple)):
        if len(path) != 1:
            raise ValueError(f"Expected a single path, got {path!r}")
        path = path[0]
    return Path(str(path)).stem


def make_annotation_result_file(*args):
    """Build an annotation-specific result path for HyperPyYAML !apply."""
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        args = tuple(args[0])
    if len(args) != 3:
        raise ValueError(
            "Expected output_folder, annotation_path, result_kind; "
            f"got {args!r}"
        )

    output_folder, annotation_path, result_kind = args
    annotation_id = get_filename_without_extension(annotation_path)
    result_kind = str(result_kind).strip("_")
    return str(Path(str(output_folder)) / f"{annotation_id}_{result_kind}.txt")
