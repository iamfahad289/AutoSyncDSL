"""
Executor/Runtime for AutoSyncDSL

This module implements the runtime that executes synchronization plans.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import bisect

from autosyncdsl.ir import (
    IR, SensorSourceNode, ExactMatchNode, NearestMatchNode,
    InterpolationNode, StaleFilterNode, BatchNode
)


@dataclass
class SensorReading:
    """A single sensor reading."""
    timestamp_ms: float
    sensor_name: str
    sensor_type: str
    data: dict  # arbitrary sensor data


@dataclass
class SyncGroup:
    """
    A synchronized group of sensor readings.
    """
    camera_readings: List[SensorReading]
    lidar_readings: List[SensorReading]
    imu_readings: List[SensorReading]
    group_timestamp: float  # primary/reference timestamp

    def __repr__(self):
        return (f"SyncGroup(ts={self.group_timestamp:.2f}, "
                f"cam={len(self.camera_readings)}, "
                f"lidar={len(self.lidar_readings)}, "
                f"imu={len(self.imu_readings)})")


class Executor:
    """
    Executes an IR plan on sensor streams.
    """

    def __init__(self, ir: IR, verbose: bool = False):
        self.ir = ir
        self.verbose = verbose
        self.logs = []
        self._timestamp_cache = {}
        self._use_timestamp_index = False

    def log(self, msg: str):
        """Log a message."""
        if self.verbose:
            print(f"[Executor] {msg}")
        self.logs.append(msg)

    def run(
        self,
        camera_stream: List[SensorReading],
        lidar_stream: List[SensorReading],
        imu_stream: List[SensorReading],
    ) -> List[SyncGroup]:
        """
        Execute the IR plan on sensor streams.

        Args:
            camera_stream: List of camera readings
            lidar_stream: List of LiDAR readings
            imu_stream: List of IMU readings

        Returns:
            List of synchronized groups
        """
        # Merge and sort all readings by timestamp
        all_readings = []
        all_readings.extend(camera_stream)
        all_readings.extend(lidar_stream)
        all_readings.extend(imu_stream)
        all_readings.sort(key=lambda r: r.timestamp_ms)

        # Extract policy from IR
        match_policy = self._extract_match_policy()
        filter_policy = self._extract_filter_policy()
        interp_config = self._extract_interp_config()
        batch_size = self._extract_batch_size()

        self.log(f"Match policy: {match_policy}")
        self.log(f"Filter policy: {filter_policy}")
        self.log(f"Interp config: {interp_config}")
        self.log(f"Batch size: {batch_size}")

        # Organize readings by sensor
        sensor_streams = self._organize_streams(
            camera_stream, lidar_stream, imu_stream
        )
        self._use_timestamp_index = bool(getattr(self.ir, "_use_timestamp_index", False))
        self._timestamp_cache = {}
        if self._use_timestamp_index:
            self._timestamp_cache = {
                id(stream): [r.timestamp_ms for r in stream]
                for stream in sensor_streams.values()
            }
        sensor_name_to_type = self._sensor_name_to_type()

        # Build synchronized groups
        groups = self._build_groups(sensor_streams, sensor_name_to_type, match_policy, interp_config)

        # Apply filtering
        filtered_groups = self._apply_filters(groups, filter_policy, sensor_streams)

        # Create batches
        batches = self._batch_groups(filtered_groups, batch_size)

        self.log(f"Built {len(groups)} initial groups, "
                 f"filtered to {len(filtered_groups)}, "
                 f"batched into {len(batches)} batches")

        return batches

    def _extract_match_policy(self) -> Dict:
        """Extract matching policy from IR."""
        policy = {
            "type": "exact",
            "primary": None,
            "tolerance_ms": 0,
            "sensors_to_match": [],
        }

        for node in self.ir.nodes.values():
            if isinstance(node, NearestMatchNode):
                policy = {
                    "type": "nearest",
                    "primary": node.primary_sensor,
                    "tolerance_ms": node.tolerance_ms,
                    "sensors_to_match": node.sensors_to_match,
                }
                break
            elif isinstance(node, ExactMatchNode):
                policy = {
                    "type": "exact",
                    "primary": node.primary_sensor,
                    "sensors_to_match": node.sensors_to_match,
                }
                break

        return policy

    def _extract_filter_policy(self) -> Optional[Dict]:
        """Extract filter policy from IR."""
        for node in self.ir.nodes.values():
            if isinstance(node, StaleFilterNode):
                return {"max_age_ms": node.max_age_ms}
        return None

    def _extract_interp_config(self) -> Dict[str, float]:
        """Extract interpolation config from IR."""
        config = {}
        for node in self.ir.nodes.values():
            if isinstance(node, InterpolationNode):
                config[node.sensor_name] = node.window_ms
        return config

    def _extract_batch_size(self) -> int:
        """Extract batch size from IR."""
        for node in self.ir.nodes.values():
            if isinstance(node, BatchNode):
                return node.batch_size
        return 1

    def _sensor_name_to_type(self) -> Dict[str, str]:
        """Map DSL sensor names to canonical stream types."""
        mapping = {}
        for node in self.ir.nodes.values():
            if isinstance(node, SensorSourceNode):
                mapping[node.sensor_name] = node.sensor_type
        return mapping

    def _organize_streams(self, cam_stream, lidar_stream, imu_stream):
        """Organize readings by sensor into indexed streams."""
        return {
            "camera": sorted(cam_stream, key=lambda r: r.timestamp_ms),
            "lidar": sorted(lidar_stream, key=lambda r: r.timestamp_ms),
            "imu": sorted(imu_stream, key=lambda r: r.timestamp_ms),
        }

    def _build_groups(
        self,
        sensor_streams: Dict[str, List[SensorReading]],
        sensor_name_to_type: Dict[str, str],
        match_policy: Dict,
        interp_config: Dict,
    ) -> List[SyncGroup]:
        """
        Build synchronized groups using the matching policy.
        """
        groups = []

        # Use the primary sensor as anchor
        primary = match_policy.get("primary", "camera")
        primary_type = sensor_name_to_type.get(primary, primary)
        primary_stream = sensor_streams.get(primary_type, [])

        if not primary_stream:
            return []

        sensors_to_match = match_policy.get("sensors_to_match", [])
        tolerance = match_policy.get("tolerance_ms", 0)
        match_type = match_policy.get("type", "exact")

        for primary_reading in primary_stream:
            group = SyncGroup(
                camera_readings=[],
                lidar_readings=[],
                imu_readings=[],
                group_timestamp=primary_reading.timestamp_ms,
            )

            # Always add primary reading
            if primary_type == "camera":
                group.camera_readings.append(primary_reading)
            elif primary_type == "lidar":
                group.lidar_readings.append(primary_reading)
            elif primary_type == "imu":
                group.imu_readings.append(primary_reading)

            # Match other sensors
            for sensor_name in sensors_to_match:
                sensor_type = sensor_name_to_type.get(sensor_name, sensor_name)
                stream = sensor_streams.get(sensor_type, [])

                if sensor_name in interp_config and sensor_type == "imu":
                    interpolated = self._interpolate_sensor(
                        primary_reading.timestamp_ms,
                        stream,
                        interp_config[sensor_name],
                    )
                    if interpolated is not None:
                        group.imu_readings.append(interpolated)
                    continue

                matched = self._find_matches(
                    primary_reading.timestamp_ms,
                    stream,
                    match_type,
                    tolerance
                )

                if matched:
                    for reading in matched:
                        if reading.sensor_type == "camera":
                            group.camera_readings.append(reading)
                        elif reading.sensor_type == "lidar":
                            group.lidar_readings.append(reading)
                        elif reading.sensor_type == "imu":
                            group.imu_readings.append(reading)
                            # Apply interpolation if configured
                            if sensor_name in interp_config:
                                # Interpolation is handled per-reading
                                pass

            groups.append(group)

        return groups

    def _find_matches(
        self,
        target_ts: float,
        stream: List[SensorReading],
        match_type: str,
        tolerance_ms: float = 0,
    ) -> List[SensorReading]:
        """
        Find matching readings in a stream for a target timestamp.

        Args:
            target_ts: Target timestamp
            stream: Stream to search
            match_type: "exact" or "nearest"
            tolerance_ms: Tolerance for matching (used with "nearest")

        Returns:
            List of matching readings
        """
        if not stream:
            return []

        if match_type == "exact":
            # Exact match: find all readings with exact timestamp
            return [r for r in stream if r.timestamp_ms == target_ts]

        else:  # nearest
            # Nearest match within tolerance. Optimized IRs reuse prebuilt
            # timestamp indexes instead of rebuilding this list per lookup.
            timestamps = self._timestamp_cache.get(id(stream))
            if timestamps is None:
                timestamps = [r.timestamp_ms for r in stream]
            idx = bisect.bisect_left(timestamps, target_ts)

            candidates = []

            if idx > 0:
                candidates.append(stream[idx - 1])
            if idx < len(stream):
                candidates.append(stream[idx])

            if not candidates:
                return []

            best = min(candidates, key=lambda r: abs(r.timestamp_ms - target_ts))
            if abs(best.timestamp_ms - target_ts) <= tolerance_ms:
                return [best]
            return []

    def _interpolate_sensor(
        self,
        target_ts: float,
        stream: List[SensorReading],
        window_ms: float,
    ) -> Optional[SensorReading]:
        """
        Linear interpolation for a high-frequency sensor stream.

        Returns an interpolated IMU reading if the target timestamp is between
        two adjacent samples and the time gap fits within the configured window.
        """
        if len(stream) < 2:
            return None

        timestamps = self._timestamp_cache.get(id(stream))
        if timestamps is None:
            timestamps = [r.timestamp_ms for r in stream]
        idx = bisect.bisect_left(timestamps, target_ts)
        if idx <= 0 or idx >= len(stream):
            return None

        left = stream[idx - 1]
        right = stream[idx]
        gap = right.timestamp_ms - left.timestamp_ms
        if gap <= 0 or gap > window_ms:
            return None

        alpha = (target_ts - left.timestamp_ms) / gap
        left_value = left.data.get("value")
        right_value = right.data.get("value")
        if not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)):
            return None

        interpolated_value = left_value + alpha * (right_value - left_value)
        return SensorReading(
            timestamp_ms=target_ts,
            sensor_name=left.sensor_name,
            sensor_type=left.sensor_type,
            data={
                "value": interpolated_value,
                "interpolated": True,
                "source_left_ts": left.timestamp_ms,
                "source_right_ts": right.timestamp_ms,
            },
        )

    def _apply_filters(
        self,
        groups: List[SyncGroup],
        filter_policy: Optional[Dict],
        sensor_streams: Dict[str, List[SensorReading]],
    ) -> List[SyncGroup]:
        """
        Apply filtering policies (e.g., stale frame rejection).
        """
        if not filter_policy:
            return groups

        max_age_ms = filter_policy.get("max_age_ms", 100)
        filtered = []

        # Keep groups only if every sensor stream has a reading close enough to the group timestamp.
        for group in groups:
            keep = True
            primary_ts = group.group_timestamp

            for sensor_type in ["camera", "lidar", "imu"]:
                stream = sensor_streams.get(sensor_type, [])
                matched = []
                if sensor_type == "camera":
                    matched = group.camera_readings
                elif sensor_type == "lidar":
                    matched = group.lidar_readings
                elif sensor_type == "imu":
                    matched = group.imu_readings

                if matched:
                    reading = matched[-1]
                    if primary_ts - reading.timestamp_ms > max_age_ms:
                        keep = False
                        break
                    continue

                prior = [r.timestamp_ms for r in stream if r.timestamp_ms <= primary_ts]
                if not prior:
                    keep = False
                    break
                if primary_ts - max(prior) > max_age_ms:
                    keep = False
                    break

            if keep:
                filtered.append(group)

        return filtered

    def _batch_groups(
        self,
        groups: List[SyncGroup],
        batch_size: int,
    ) -> List[List[SyncGroup]]:
        """
        Create fixed-size batches of synchronized groups.
        """
        batches = []
        for i in range(0, len(groups), batch_size):
            batch = groups[i:i+batch_size]
            if batch:
                batches.append(batch)
        return batches
