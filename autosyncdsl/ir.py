"""
Intermediate Representation (IR) for AutoSyncDSL

This module defines the internal representation of synchronization plans.
The IR is a simple, inspectable graph of nodes representing:
- Sensor sources
- Matching policies
- Filtering operations
- Interpolation operations
- Batching
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class IRNode:
    """Base class for IR nodes."""
    node_id: str
    node_type: str

    def __repr__(self):
        return f"{self.node_type}({self.node_id})"


@dataclass
class SensorSourceNode(IRNode):
    """Represents a sensor data source."""
    sensor_type: str  # "camera", "lidar", "imu"
    sensor_name: str
    sample_rate_hz: Optional[float] = None

    def __init__(self, sensor_name: str, sensor_type: str):
        super().__init__(
            node_id=f"source_{sensor_name}",
            node_type="sensor_source"
        )
        self.sensor_name = sensor_name
        self.sensor_type = sensor_type


@dataclass
class ExactMatchNode(IRNode):
    """Exact timestamp matching between sensors."""
    primary_sensor: str
    sensors_to_match: List[str] = field(default_factory=list)

    def __init__(self, node_id: str, primary: str, to_match: List[str]):
        super().__init__(node_id, "exact_match")
        self.primary_sensor = primary
        self.sensors_to_match = to_match


@dataclass
class NearestMatchNode(IRNode):
    """Nearest-neighbor matching within tolerance."""
    primary_sensor: str
    sensors_to_match: List[str] = field(default_factory=list)
    tolerance_ms: float = 50.0

    def __init__(self, node_id: str, primary: str, to_match: List[str], tol: float):
        super().__init__(node_id, "nearest_match")
        self.primary_sensor = primary
        self.sensors_to_match = to_match
        self.tolerance_ms = tol


@dataclass
class InterpolationNode(IRNode):
    """Interpolation for high-frequency sensors (e.g., IMU)."""
    sensor_name: str
    window_ms: float = 20.0

    def __init__(self, node_id: str, sensor: str, window: float):
        super().__init__(node_id, "interpolation")
        self.sensor_name = sensor
        self.window_ms = window


@dataclass
class StaleFilterNode(IRNode):
    """Filter out stale frames."""
    max_age_ms: float = 100.0

    def __init__(self, node_id: str, max_age: float):
        super().__init__(node_id, "stale_filter")
        self.max_age_ms = max_age


@dataclass
class BatchNode(IRNode):
    """Batch construction node."""
    batch_size: int = 4

    def __init__(self, node_id: str, size: int):
        super().__init__(node_id, "batch")
        self.batch_size = size


class IR:
    """
    Intermediate representation of a synchronization plan.

    Stores nodes and the execution flow as a DAG.
    """

    def __init__(self):
        self.nodes: Dict[str, IRNode] = {}
        self.edges: List[tuple] = []  # (from_id, to_id)
        self.sources: List[str] = []  # sensor source node IDs
        self.root_nodes: List[str] = []  # execution roots

    def add_node(self, node: IRNode) -> None:
        """Add a node to the IR."""
        self.nodes[node.node_id] = node
        if isinstance(node, SensorSourceNode):
            self.sources.append(node.node_id)

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add an edge in the execution DAG."""
        self.edges.append((from_id, to_id))

    def set_roots(self, root_ids: List[str]) -> None:
        """Set the root nodes for execution."""
        self.root_nodes = root_ids

    def __repr__(self):
        lines = ["IR Plan:"]
        lines.append(f"  Nodes: {len(self.nodes)}")
        for node in self.nodes.values():
            lines.append(f"    {node}")
        lines.append(f"  Edges: {len(self.edges)}")
        for src, dst in self.edges:
            lines.append(f"    {src} -> {dst}")
        lines.append(f"  Roots: {self.root_nodes}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize IR to dictionary for inspection."""
        nodes_dict = {}
        for node_id, node in self.nodes.items():
            node_dict = {
                "type": node.node_type,
                "node_id": node.node_id,
            }
            if isinstance(node, SensorSourceNode):
                node_dict.update({
                    "sensor_type": node.sensor_type,
                    "sensor_name": node.sensor_name,
                })
            elif isinstance(node, ExactMatchNode):
                node_dict.update({
                    "primary_sensor": node.primary_sensor,
                    "sensors_to_match": node.sensors_to_match,
                })
            elif isinstance(node, NearestMatchNode):
                node_dict.update({
                    "primary_sensor": node.primary_sensor,
                    "sensors_to_match": node.sensors_to_match,
                    "tolerance_ms": node.tolerance_ms,
                })
            elif isinstance(node, InterpolationNode):
                node_dict.update({
                    "sensor_name": node.sensor_name,
                    "window_ms": node.window_ms,
                })
            elif isinstance(node, StaleFilterNode):
                node_dict["max_age_ms"] = node.max_age_ms
            elif isinstance(node, BatchNode):
                node_dict["batch_size"] = node.batch_size
            nodes_dict[node_id] = node_dict

        return {
            "nodes": nodes_dict,
            "edges": self.edges,
            "sources": self.sources,
            "roots": self.root_nodes,
        }

