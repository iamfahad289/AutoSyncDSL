"""
Utility functions for AutoSyncDSL
"""

import json
from typing import Any, Dict, List


def ir_to_json(ir: "IR") -> str:
    """Serialize IR to JSON for inspection."""
    return json.dumps(ir.to_dict(), indent=2)


def print_ir(ir: "IR") -> None:
    """Pretty-print IR."""
    print(ir)


def print_batch_results(batches: List[List[Any]]) -> None:
    """Print batch results nicely."""
    for i, batch in enumerate(batches):
        print(f"\nBatch {i}:")
        for j, group in enumerate(batch):
            print(f"  Group {j}: {group}")

