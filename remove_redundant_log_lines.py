#!/usr/bin/env python3
"""
Remove redundant tqdm progress-bar lines from eval log files.

Removes lines like:
  Evaluating:   1%|...| 35/5000 [01:08<2:37:19,  1.90s/it]
  Classes:  12%|...| 4/34 [00:25<03:14,  6.49s/it]

Optionally keeps the final line of each progress block (e.g. 100% ... 5000/5000).

Usage:
  python remove_redundant_log_lines.py "eval_log_20260127_223503 (2).txt"
  python remove_redundant_log_lines.py "eval_log_20260127_223503 (2).txt" --keep-final
  python remove_redundant_log_lines.py "eval_log_20260127_223503 (2).txt" -o cleaned_log.txt
"""

import re
import argparse
import os
from typing import Optional


def is_tqdm_progress_line(line: str) -> bool:
    """True if line looks like a tqdm progress bar (Evaluating: N%|...| M/TOTAL [..., Xs/it])."""
    # Must contain percentage-style progress and tqdm ETA (e.g. "s/it]" or "?it/s]")
    if "s/it]" not in line and "it/s]" not in line:
        return False
    # Common tqdm labels: "Evaluating:   0%|" or "Classes:  12%|" (label has colon, then spaces, then N%)
    if re.search(r"(Evaluating|Classes|Loading|Saving|Processing):\s*\d+%\s*\|", line):
        return True
    # Generic: "Label:  N%|" ... "| M/TOTAL ["
    if re.search(r":\s*\d+%\s*\|", line) and re.search(r"\|\s*\d+/\d+\s+\[", line):
        return True
    return False


def is_final_tqdm_line(line: str) -> bool:
    """True if line is a 100% completion line (e.g. 5000/5000)."""
    if not is_tqdm_progress_line(line):
        return False
    # 100%|...| TOTAL/TOTAL [...]
    return "100%|" in line or re.search(r"\|\s*(\d+)/\1\s+\[", line) is not None


def clean_log(
    input_path: str,
    output_path: Optional[str] = None,
    keep_final_per_block: bool = False,
    in_place: bool = False,
) -> None:
    """
    Read log file, remove redundant progress lines, write result.

    If output_path is None and not in_place: writes to {input_path}_cleaned.txt
    If in_place: overwrites input (use with backup).
    """
    input_path = os.path.abspath(input_path)
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)

    if output_path is None:
        if in_place:
            output_path = input_path
        else:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_cleaned{ext}"

    with open(input_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    if in_place and output_path == input_path:
        # Write to temp then replace
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".txt", prefix="eval_log_clean_")
        try:
            out = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            os.close(fd)
            os.remove(tmp)
            raise
    else:
        tmp = None
        out = open(output_path, "w", encoding="utf-8")

    removed = 0
    try:
        for line in lines:
            if not is_tqdm_progress_line(line):
                out.write(line)
                continue
            # It's a tqdm line
            if keep_final_per_block and is_final_tqdm_line(line):
                out.write(line)
            else:
                removed += 1
    finally:
        out.close()
        if tmp is not None:
            os.replace(tmp, input_path)

    print(f"Removed {removed} redundant progress line(s).")
    if not in_place and output_path != input_path:
        print(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove redundant tqdm progress lines from eval log files."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="eval_log_20260127_223503 (2).txt",
        help="Input log file path",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output file path (default: <input>_cleaned.txt)",
    )
    parser.add_argument(
        "--keep-final",
        action="store_true",
        help="Keep the final (100% / N/N) line of each progress block",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file (redundant lines removed in place)",
    )
    args = parser.parse_args()

    clean_log(
        args.input,
        output_path=args.output,
        keep_final_per_block=args.keep_final,
        in_place=args.in_place,
    )


if __name__ == "__main__":
    main()
