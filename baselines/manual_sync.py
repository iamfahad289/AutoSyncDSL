"""
Handwritten Baseline Implementation for Multi-Sensor Synchronization

This is a direct, imperative implementation of the same synchronization logic
without using the DSL, IR, or executor abstraction.

Used for comparison with the DSL-based approach.
"""

from typing import List, Tuple, Dict
from collections import defaultdict
import bisect

from autosyncdsl.executor import SensorReading, SyncGroup


class ManualSyncBaseline:
    """
    Handwritten baseline for multi-sensor synchronization.

    Implements the core logic directly without abstraction layers.
    """

    def __init__(
        self,
        match_type: str = "nearest",
        tolerance_ms: float = 50.0,
        max_age_ms: float = 100.0,
        batch_size: int = 4,
        interpolate_imu: bool = False,
        interpolation_window_ms: float = 120.0,
    ):
        """
        Initialize baseline with configuration.

        Args:
            match_type: "exact" or "nearest"
            tolerance_ms: Tolerance for nearest matching
            max_age_ms: Maximum age for stale filtering
            batch_size: Batch size
        """
        self.match_type = match_type
        self.tolerance_ms = tolerance_ms
        self.max_age_ms = max_age_ms
        self.batch_size = batch_size
        self.interpolate_imu = interpolate_imu
        self.interpolation_window_ms = interpolation_window_ms

    def synchronize(
        self,
        camera_stream: List[SensorReading],
        lidar_stream: List[SensorReading],
        imu_stream: List[SensorReading],
    ) -> List[List[SyncGroup]]:
        """
        Synchronize three sensor streams using direct imperative logic.

        Returns:
            List of batches, where each batch is a list of SyncGroup
        """
        # Step 1: Sort all streams
        camera_stream = sorted(camera_stream, key=lambda r: r.timestamp_ms)
        lidar_stream = sorted(lidar_stream, key=lambda r: r.timestamp_ms)
        imu_stream = sorted(imu_stream, key=lambda r: r.timestamp_ms)

        # Step 2: Use camera as primary stream (anchor)
        groups = []

        for cam_reading in camera_stream:
            # Create a group with this camera reading
            group = SyncGroup(
                camera_readings=[cam_reading],
                lidar_readings=[],
                imu_readings=[],
                group_timestamp=cam_reading.timestamp_ms,
            )

            # Find matching LiDAR reading
            lidar_matches = self._find_matching_readings(
                cam_reading.timestamp_ms,
                lidar_stream,
            )
            group.lidar_readings.extend(lidar_matches)

            if self.interpolate_imu:
                interpolated = self._interpolate_imu(cam_reading.timestamp_ms, imu_stream)
                if interpolated is not None:
                    group.imu_readings.append(interpolated)
            else:
                imu_matches = self._find_matching_readings(
                    cam_reading.timestamp_ms,
                    imu_stream,
                )
                group.imu_readings.extend(imu_matches)

            groups.append(group)

        # Step 3: Apply stale filtering
        filtered_groups = []
        for group in groups:
            if self._is_not_stale(group, camera_stream, lidar_stream, imu_stream):
                filtered_groups.append(group)

        # Step 4: Create batches
        batches = []
        for i in range(0, len(filtered_groups), self.batch_size):
            batch = filtered_groups[i:i + self.batch_size]
            if batch:
                batches.append(batch)

        return batches

    def _find_matching_readings(
        self,
        target_ts: float,
        stream: List[SensorReading],
    ) -> List[SensorReading]:
        """
        Find matching readings for a target timestamp.

        Implements the matching policy (exact or nearest).
        """
        if not stream:
            return []

        if self.match_type == "exact":
            return [r for r in stream if r.timestamp_ms == target_ts]

        # nearest
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
        if abs(best.timestamp_ms - target_ts) <= self.tolerance_ms:
            return [best]
        return []

    def _interpolate_imu(
        self,
        target_ts: float,
        stream: List[SensorReading],
    ) -> SensorReading | None:
        """Interpolate IMU value using the same bracketing rule as the DSL runtime."""
        if len(stream) < 2:
            return None

        timestamps = [r.timestamp_ms for r in stream]
        idx = bisect.bisect_left(timestamps, target_ts)
        if idx <= 0 or idx >= len(stream):
            return None

        left = stream[idx - 1]
        right = stream[idx]
        gap = right.timestamp_ms - left.timestamp_ms
        if gap <= 0 or gap > self.interpolation_window_ms:
            return None

        left_value = left.data.get("value")
        right_value = right.data.get("value")
        if not isinstance(left_value, (int, float)) or not isinstance(right_value, (int, float)):
            return None

        alpha = (target_ts - left.timestamp_ms) / gap
        return SensorReading(
            timestamp_ms=target_ts,
            sensor_name=left.sensor_name,
            sensor_type=left.sensor_type,
            data={
                "value": left_value + alpha * (right_value - left_value),
                "interpolated": True,
                "source_left_ts": left.timestamp_ms,
                "source_right_ts": right.timestamp_ms,
            },
        )

    def _is_not_stale(
        self,
        group: SyncGroup,
        camera_stream: List[SensorReading],
        lidar_stream: List[SensorReading],
        imu_stream: List[SensorReading],
    ) -> bool:
        """
        Check if a group satisfies stale filtering policy.

        A group is not stale if all matched readings are within max_age_ms of
        primary and any missing stream has no prior stale reading that should
        have been rejected. This mirrors the DSL executor's stale semantics.
        """
        primary_ts = group.group_timestamp

        streams = {
            "camera": camera_stream,
            "lidar": lidar_stream,
            "imu": imu_stream,
        }
        matched_by_type = {
            "camera": group.camera_readings,
            "lidar": group.lidar_readings,
            "imu": group.imu_readings,
        }

        for sensor_type, stream in streams.items():
            matched = matched_by_type[sensor_type]
            if matched:
                reading = matched[-1]
                if primary_ts - reading.timestamp_ms > self.max_age_ms:
                    return False
                continue

            prior_timestamps = [r.timestamp_ms for r in stream if r.timestamp_ms <= primary_ts]
            if not prior_timestamps:
                return False
            if primary_ts - max(prior_timestamps) > self.max_age_ms:
                return False

        return True


class ManualSyncVariant2:
    """
    Variant of baseline using different approach.

    Uses sliding window and event-driven matching instead of
    primary-stream anchoring.
    """

    def __init__(
        self,
        tolerance_ms: float = 50.0,
        max_age_ms: float = 100.0,
        batch_size: int = 4,
    ):
        self.tolerance_ms = tolerance_ms
        self.max_age_ms = max_age_ms
        self.batch_size = batch_size

    def synchronize(
        self,
        camera_stream: List[SensorReading],
        lidar_stream: List[SensorReading],
        imu_stream: List[SensorReading],
    ) -> List[List[SyncGroup]]:
        """
        Synchronize using event-driven approach.
        """
        # Merge all readings with sensor type annotation
        events = []
        for r in camera_stream:
            events.append(("camera", r))
        for r in lidar_stream:
            events.append(("lidar", r))
        for r in imu_stream:
            events.append(("imu", r))

        # Sort by timestamp
        events.sort(key=lambda e: e[1].timestamp_ms)

        # Group nearby events
        groups = []
        current_group = None
        last_ts = None

        for sensor_type, reading in events:
            # Start new group if gap is too large
            if last_ts is None or reading.timestamp_ms - last_ts > self.tolerance_ms * 2:
                if current_group and self._group_is_complete(current_group):
                    groups.append(current_group)
                current_group = SyncGroup(
                    camera_readings=[],
                    lidar_readings=[],
                    imu_readings=[],
                    group_timestamp=reading.timestamp_ms,
                )

            # Add reading to current group
            if sensor_type == "camera":
                current_group.camera_readings.append(reading)
            elif sensor_type == "lidar":
                current_group.lidar_readings.append(reading)
            elif sensor_type == "imu":
                current_group.imu_readings.append(reading)

            last_ts = reading.timestamp_ms

        # Don't forget final group
        if current_group and self._group_is_complete(current_group):
            groups.append(current_group)

        # Filter and batch
        filtered = [g for g in groups if not self._is_stale(g)]

        batches = []
        for i in range(0, len(filtered), self.batch_size):
            batch = filtered[i:i + self.batch_size]
            if batch:
                batches.append(batch)

        return batches

    def _group_is_complete(self, group: SyncGroup) -> bool:
        """Check if group has readings from all sensors."""
        return (len(group.camera_readings) > 0 and
                len(group.lidar_readings) > 0 and
                len(group.imu_readings) > 0)

    def _is_stale(self, group: SyncGroup) -> bool:
        """Check if group is stale."""
        primary_ts = group.group_timestamp

        for reading in (group.camera_readings + group.lidar_readings +
                       group.imu_readings):
            if reading.timestamp_ms < primary_ts:
                age = primary_ts - reading.timestamp_ms
                if age > self.max_age_ms:
                    return True

        return False


class SimpleSync:
    """
    Simplest possible synchronization: just collect readings within time windows.

    For performance comparison.
    """

    def __init__(self, window_ms: float = 100.0, batch_size: int = 4):
        self.window_ms = window_ms
        self.batch_size = batch_size

    def synchronize(
        self,
        camera_stream: List[SensorReading],
        lidar_stream: List[SensorReading],
        imu_stream: List[SensorReading],
    ) -> List[List[SyncGroup]]:
        """
        Simple time-window based synchronization.
        """
        # Use camera as primary
        camera_stream = sorted(camera_stream, key=lambda r: r.timestamp_ms)

        # Index other streams for fast lookup
        lidar_by_ts = sorted(lidar_stream, key=lambda r: r.timestamp_ms)
        imu_by_ts = sorted(imu_stream, key=lambda r: r.timestamp_ms)

        groups = []

        for cam_reading in camera_stream:
            window_start = cam_reading.timestamp_ms - self.window_ms / 2
            window_end = cam_reading.timestamp_ms + self.window_ms / 2

            # Collect readings in window
            lidar_in_window = [
                r for r in lidar_by_ts
                if window_start <= r.timestamp_ms <= window_end
            ]

            imu_in_window = [
                r for r in imu_by_ts
                if window_start <= r.timestamp_ms <= window_end
            ]

            group = SyncGroup(
                camera_readings=[cam_reading],
                lidar_readings=lidar_in_window,
                imu_readings=imu_in_window,
                group_timestamp=cam_reading.timestamp_ms,
            )

            groups.append(group)

        # Batch
        batches = []
        for i in range(0, len(groups), self.batch_size):
            batch = groups[i:i + self.batch_size]
            if batch:
                batches.append(batch)

        return batches
