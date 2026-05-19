"""
Fluent Embedded Python DSL for AutoSyncDSL

This module provides the user-facing API for building synchronization plans.
"""

from typing import List, Optional
from autosyncdsl.ir import (
    IR, SensorSourceNode, ExactMatchNode, NearestMatchNode,
    InterpolationNode, StaleFilterNode, BatchNode
)


class SyncPlan:
    """
    Fluent builder for synchronization plans.

    Example:
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .nearest("lidar", tolerance_ms=50)
              .interpolate("imu", window_ms=20)
              .drop_stale(max_age_ms=100)
              .batch(size=4)
        )
    """

    def __init__(self):
        self.ir = IR()
        self.sensors: dict = {}  # sensor_name -> sensor_type
        self.match_ops: list = []  # operations to apply
        self.filter_ops: list = []
        self.interp_ops: list = []
        self.batch_config: Optional[dict] = None
        self.node_counter = 0

    def camera(self, name: str) -> "SyncPlan":
        """Register a camera sensor."""
        self.sensors[name] = "camera"
        return self

    def lidar(self, name: str) -> "SyncPlan":
        """Register a LiDAR sensor."""
        self.sensors[name] = "lidar"
        return self

    def imu(self, name: str) -> "SyncPlan":
        """Register an IMU sensor."""
        self.sensors[name] = "imu"
        return self

    def exact(self, primary: str, *others: str) -> "SyncPlan":
        """
        Exact timestamp matching.

        Args:
            primary: Primary sensor to match on
            others: Other sensors to match to primary (if empty, use all others)
        """
        # If no others specified, match all registered sensors except primary
        match_others = list(others) if others else [n for n in self.sensors.keys() if n != primary]

        self.match_ops.append({
            "type": "exact",
            "primary": primary,
            "others": match_others,
        })
        return self

    def nearest(self, primary: str, tolerance_ms: float = 50.0, *others: str) -> "SyncPlan":
        """
        Nearest-neighbor matching within tolerance.

        Args:
            primary: Primary sensor to match on
            tolerance_ms: Tolerance in milliseconds
            others: Other sensors to match to primary (if empty, use all others)
        """
        # If no others specified, match all registered sensors except primary
        match_others = list(others) if others else [n for n in self.sensors.keys() if n != primary]

        self.match_ops.append({
            "type": "nearest",
            "primary": primary,
            "tolerance_ms": tolerance_ms,
            "others": match_others,
        })
        return self

    def interpolate(self, sensor: str, window_ms: float = 20.0) -> "SyncPlan":
        """
        Enable interpolation for a sensor (typically IMU).

        Args:
            sensor: Sensor to interpolate
            window_ms: Interpolation window in milliseconds
        """
        self.interp_ops.append({
            "sensor": sensor,
            "window_ms": window_ms,
        })
        return self

    def drop_stale(self, max_age_ms: float = 100.0) -> "SyncPlan":
        """
        Filter out stale frames.

        Args:
            max_age_ms: Maximum age of a frame in milliseconds
        """
        self.filter_ops.append({
            "type": "stale_filter",
            "max_age_ms": max_age_ms,
        })
        return self

    def batch(self, size: int = 4) -> "SyncPlan":
        """
        Configure batch size.

        Args:
            size: Number of synchronized groups per batch
        """
        self.batch_config = {"size": size}
        return self

    def compile(self) -> IR:
        """
        Compile the plan to an intermediate representation.

        Returns:
            IR object ready for execution
        """
        self.ir = IR()

        # Add sensor sources
        for sensor_name, sensor_type in self.sensors.items():
            node = SensorSourceNode(sensor_name, sensor_type)
            self.ir.add_node(node)

        node_id = 0
        prev_nodes = list(self.ir.sources)

        # Add matching nodes
        for match_op in self.match_ops:
            node_id_str = f"match_{node_id}"
            node_id += 1

            if match_op["type"] == "exact":
                node = ExactMatchNode(
                    node_id_str,
                    match_op["primary"],
                    match_op["others"]
                )
            else:  # nearest
                node = NearestMatchNode(
                    node_id_str,
                    match_op["primary"],
                    match_op["others"],
                    match_op["tolerance_ms"]
                )

            self.ir.add_node(node)
            for prev in prev_nodes:
                self.ir.add_edge(prev, node_id_str)
            prev_nodes = [node_id_str]

        # Add filter nodes
        for filter_op in self.filter_ops:
            node_id_str = f"filter_{node_id}"
            node_id += 1

            node = StaleFilterNode(node_id_str, filter_op["max_age_ms"])
            self.ir.add_node(node)
            for prev in prev_nodes:
                self.ir.add_edge(prev, node_id_str)
            prev_nodes = [node_id_str]

        # Add interpolation nodes
        for interp_op in self.interp_ops:
            node_id_str = f"interp_{node_id}"
            node_id += 1

            node = InterpolationNode(
                node_id_str,
                interp_op["sensor"],
                interp_op["window_ms"]
            )
            self.ir.add_node(node)
            for prev in prev_nodes:
                self.ir.add_edge(prev, node_id_str)
            prev_nodes = [node_id_str]

        # Add batch node
        if self.batch_config:
            node_id_str = f"batch_{node_id}"
            node = BatchNode(node_id_str, self.batch_config["size"])
            self.ir.add_node(node)
            for prev in prev_nodes:
                self.ir.add_edge(prev, node_id_str)
            prev_nodes = [node_id_str]

        # Set roots
        self.ir.set_roots(prev_nodes)

        return self.ir

