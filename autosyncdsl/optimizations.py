"""
Optimization Passes for AutoSyncDSL

This module implements two lightweight optimization passes:
1. Rule Fusion: Combine compatible operations
2. Buffer Reuse: Reuse temporary data structures
"""

from typing import Dict, List, Tuple
from autosyncdsl.ir import (
    IR, IRNode, NearestMatchNode, ExactMatchNode,
    InterpolationNode, StaleFilterNode, BatchNode
)


class Optimizer:
    """
    Applies optimization passes to IR.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logs = []

    def log(self, msg: str):
        """Log optimization step."""
        if self.verbose:
            print(f"[Optimizer] {msg}")
        self.logs.append(msg)

    def optimize(self, ir: IR) -> IR:
        """
        Apply optimization passes to IR.

        Args:
            ir: Input IR

        Returns:
            Optimized IR
        """
        self.log("Starting optimization passes")

        # Pass 1: Rule fusion
        ir = self.fuse_rules(ir)

        # Pass 2: Buffer reuse (preparation - documented but not exec change)
        ir = self.mark_buffer_reuse(ir)

        # Pass 3: timestamp index reuse for binary-search matching and
        # interpolation lookups. The executor consumes this annotation.
        ir._use_timestamp_index = True
        ir._optimization_pass_count = getattr(ir, '_optimization_pass_count', 0) + 1

        self.log("Optimization complete")
        return ir

    def fuse_rules(self, ir: IR) -> IR:
        """
        Fusion Pass: Combine compatible adjacent nodes.

        For example, combine:
        - ExactMatch -> Interpolation -> StaleFilter
        into a single fused operation.

        This reduces node traversal overhead.
        """
        self.log("Applying rule fusion pass")

        # Analyze the execution graph
        node_list = []
        for root in ir.root_nodes:
            node_list.extend(self._collect_ancestors(ir, root))

        # Look for fusion opportunities
        # Strategy: Find chains of compatible operations
        fusion_candidates = self._find_fusion_opportunities(ir)

        if fusion_candidates:
            self.log(f"Found {len(fusion_candidates)} fusion opportunities")
            for chain in fusion_candidates:
                self.log(f"  Can fuse chain: {' -> '.join(chain)}")

        # For now, fusion is annotated in the IR (not physically changing structure)
        # In a real compiler, we'd create fused nodes
        ir._fusion_candidates = fusion_candidates
        ir._optimization_pass_count = getattr(ir, '_optimization_pass_count', 0) + 1

        return ir

    def mark_buffer_reuse(self, ir: IR) -> IR:
        """
        Buffer Reuse Pass: Identify opportunities for reusing temporary buffers.

        Strategy: Identify nodes that produce similar intermediate results
        and can share buffer space if their lifetimes don't overlap.

        This reduces memory allocation overhead.
        """
        self.log("Applying buffer reuse pass")

        # Analyze lifetime of each node's output
        buffer_candidates = self._analyze_buffer_lifetimes(ir)

        if buffer_candidates:
            self.log(f"Found {len(buffer_candidates)} buffer reuse opportunities")
            for candidate in buffer_candidates:
                self.log(f"  Can reuse buffer for: {candidate}")

        # Annotate IR with reuse info
        ir._buffer_reuse_candidates = buffer_candidates
        ir._optimization_pass_count = getattr(ir, '_optimization_pass_count', 0) + 1

        return ir

    def _collect_ancestors(self, ir: IR, node_id: str, visited=None) -> List[str]:
        """Collect all ancestor nodes of a given node."""
        if visited is None:
            visited = set()

        if node_id in visited:
            return []

        visited.add(node_id)
        ancestors = [node_id]

        # Find predecessors
        for src, dst in ir.edges:
            if dst == node_id:
                ancestors.extend(self._collect_ancestors(ir, src, visited))

        return ancestors

    def _find_fusion_opportunities(self, ir: IR) -> List[List[str]]:
        """
        Find chains of nodes that can be fused together.

        Fusible patterns:
        - Match -> Interpolation
        - Interpolation -> StaleFilter
        - Match -> StaleFilter
        - Any sequence where output of one directly feeds into another
        """
        chains = []

        # Build a dependency graph for analysis
        predecessors = {}
        successors = {}

        for src, dst in ir.edges:
            if dst not in predecessors:
                predecessors[dst] = []
            predecessors[dst].append(src)

            if src not in successors:
                successors[src] = []
            successors[src].append(dst)

        # Find maximal chains
        visited = set()
        for node_id in ir.nodes:
            if node_id in visited:
                continue

            # Check if this is a chain start (has no predecessors or multiple)
            preds = predecessors.get(node_id, [])
            if len(preds) == 1:
                continue  # Part of a longer chain

            # Trace forward from this node
            chain = [node_id]
            current = node_id
            visited.add(current)

            while current in successors and len(successors[current]) == 1:
                next_node = successors[current][0]

                # Check if next node is fusible with current
                if self._is_fusible(ir.nodes[current], ir.nodes[next_node]):
                    chain.append(next_node)
                    visited.add(next_node)
                    current = next_node
                else:
                    break

            if len(chain) > 1:
                chains.append(chain)

        return chains

    def _is_fusible(self, node1: IRNode, node2: IRNode) -> bool:
        """Check if two adjacent nodes can be fused."""
        # Define fusible patterns
        type1 = node1.node_type
        type2 = node2.node_type

        fusible_patterns = [
            ("exact_match", "interpolation"),
            ("exact_match", "stale_filter"),
            ("nearest_match", "interpolation"),
            ("nearest_match", "stale_filter"),
            ("interpolation", "stale_filter"),
            ("stale_filter", "batch"),
            ("interpolation", "batch"),
        ]

        return (type1, type2) in fusible_patterns

    def _analyze_buffer_lifetimes(self, ir: IR) -> List[str]:
        """
        Analyze which nodes produce buffers that can be reused.

        Returns node IDs that are candidates for buffer reuse.
        """
        reuse_candidates = []

        # Nodes that produce large intermediate buffers:
        # - Matching nodes (produce lists of matched readings)
        # - Interpolation nodes (produce interpolated data)

        for node_id, node in ir.nodes.items():
            if node.node_type in ["exact_match", "nearest_match", "interpolation"]:
                reuse_candidates.append(node_id)

        return reuse_candidates


class BufferPool:
    """
    Simple buffer pool for reusing temporary data structures.

    In the executor, this can reduce allocation overhead.
    """

    def __init__(self):
        self.pools = {}

    def get_buffer(self, buffer_type: str, size: int):
        """
        Get a reusable buffer, or allocate a new one.

        Args:
            buffer_type: Type of buffer ("list", "dict", etc.)
            size: Expected size

        Returns:
            A reusable buffer
        """
        key = (buffer_type, size)

        if key not in self.pools:
            self.pools[key] = []

        if self.pools[key]:
            return self.pools[key].pop()

        # Allocate new buffer
        if buffer_type == "list":
            return []
        elif buffer_type == "dict":
            return {}
        else:
            return None

    def return_buffer(self, buffer_type: str, size: int, buffer):
        """
        Return a buffer to the pool for reuse.

        Args:
            buffer_type: Type of buffer
            size: Size
            buffer: Buffer to return
        """
        key = (buffer_type, size)

        if key not in self.pools:
            self.pools[key] = []

        # Clear the buffer before returning
        if isinstance(buffer, list):
            buffer.clear()
        elif isinstance(buffer, dict):
            buffer.clear()

        self.pools[key].append(buffer)
