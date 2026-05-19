"""
KITTI Dataset Loader for AutoSyncDSL

Loads a real KITTI Raw drive if available, provides fallback interface.
"""

import os
from datetime import datetime
from typing import List, Tuple

from autosyncdsl.executor import SensorReading


class KITTILoader:
    """
    Load KITTI Raw data subset.

    Expected KITTI structure:
    data/raw/kitti/
    └── 2011_09_26/
        ├── calib_cam_to_cam.txt
        ├── calib_imu_to_velo.txt
        ├── calib_velo_to_cam.txt
        └── 2011_09_26_drive_0001_sync/
            ├── image_00/
            ├── image_01/
            ├── image_02/
            ├── image_03/
            ├── velodyne_points/
            └── oxts/
    """

    def __init__(self, kitti_root: str = "data/raw/kitti"):
        self.kitti_root = kitti_root
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        """Check if KITTI data is available."""
        return os.path.isdir(self.kitti_root)

    def load_drives(self) -> List[str]:
        """List available KITTI drives."""
        if not self.available:
            return []

        drives = []
        for root, dirs, _files in os.walk(self.kitti_root):
            for d in dirs:
                if d.endswith("_sync") and d.startswith("2011_"):
                    rel = os.path.relpath(os.path.join(root, d), self.kitti_root)
                    drives.append(rel)

        return sorted(drives)

    def load_sequence(
        self,
        drive_name: str,
        max_frames: int | None = 100,
    ) -> Tuple[List[SensorReading], List[SensorReading], List[SensorReading]]:
        """
        Load a sequence from KITTI.

        Args:
            drive_name: Name of the drive (e.g., "2011_09_26_drive_0001_sync")
            max_frames: Maximum frames to load

        Returns:
            Tuple of (camera_stream, lidar_stream, imu_stream)
        """
        drive_path = self._resolve_drive_path(drive_name)

        camera_ts = self._load_timestamps(os.path.join(drive_path, "image_02", "timestamps.txt"))
        lidar_ts = self._load_timestamps(os.path.join(drive_path, "velodyne_points", "timestamps.txt"))
        imu_ts = self._load_timestamps(os.path.join(drive_path, "oxts", "timestamps.txt"))

        if not camera_ts or not lidar_ts or not imu_ts:
            return [], [], []

        if max_frames is not None:
            camera_ts = camera_ts[:max_frames]
            lidar_ts = lidar_ts[:max_frames]
            imu_ts = imu_ts[:max_frames]

        camera_stream = [
            SensorReading(
                timestamp_ms=self._timestamp_to_ms(ts),
                sensor_name="camera",
                sensor_type="camera",
                data={"frame_idx": i, "timestamp": ts},
            )
            for i, ts in enumerate(camera_ts)
        ]

        lidar_stream = [
            SensorReading(
                timestamp_ms=self._timestamp_to_ms(ts),
                sensor_name="lidar",
                sensor_type="lidar",
                data={"frame_idx": i, "timestamp": ts},
            )
            for i, ts in enumerate(lidar_ts)
        ]

        imu_stream = [
            SensorReading(
                timestamp_ms=self._timestamp_to_ms(ts),
                sensor_name="imu",
                sensor_type="imu",
                data={"frame_idx": i, "timestamp": ts},
            )
            for i, ts in enumerate(imu_ts)
        ]

        return camera_stream, lidar_stream, imu_stream

    def _resolve_drive_path(self, drive_name: str) -> str:
        """Resolve a drive name relative to the KITTI root."""
        direct = os.path.join(self.kitti_root, drive_name)
        if os.path.isdir(direct):
            return direct

        nested = os.path.join(self.kitti_root, "2011_09_26", drive_name)
        if os.path.isdir(nested):
            return nested

        for root, dirs, _files in os.walk(self.kitti_root):
            if drive_name in dirs:
                return os.path.join(root, drive_name)

        raise FileNotFoundError("KITTI drive not found: %s" % drive_name)

    def _load_timestamps(self, timestamps_file: str) -> List[str]:
        """Load timestamps for a drive."""
        if os.path.exists(timestamps_file):
            with open(timestamps_file, "r") as f:
                return [line.strip() for line in f if line.strip()]

        return []

    def _timestamp_to_ms(self, timestamp_str: str) -> float:
        """Convert KITTI timestamp string to milliseconds since epoch-like reference."""
        dt_part, frac_part = timestamp_str.split(".")
        base = datetime.strptime(dt_part, "%Y-%m-%d %H:%M:%S")
        frac = (frac_part + "000000000")[:9]
        seconds = base.timestamp()
        return seconds * 1000.0 + int(frac[:6]) / 1000.0

    @staticmethod
    def is_kitti_available(kitti_root: str = "data/raw/kitti") -> bool:
        """Check if KITTI data is available."""
        return os.path.isdir(kitti_root) and bool(KITTILoader(kitti_root).load_drives())


class FallbackKITTIInterface:
    """
    Fallback interface when KITTI is not available.

    Provides the same interface but generates perturbed KITTI data.
    """

    def __init__(self):
        pass

    def load_drives(self) -> List[str]:
        """Return empty list (no real drives available)."""
        return []

    def load_sequence(
        self,
        drive_name: str,
        max_frames: int = 100,
    ) -> Tuple[List[SensorReading], List[SensorReading], List[SensorReading]]:
        """
        Return perturbed KITTI data in KITTI-like format.
        """
        from loaders.kitti_perturbation import KITTIPerturbationGenerator

        gen = KITTIPerturbationGenerator(
            duration_sec=10.0,
            camera_rate_hz=30.0,
            lidar_rate_hz=10.0,
            imu_rate_hz=100.0,
        )

        return gen.generate(jitter_ms=5.0, dropout_rate=0.01)


def get_kitti_loader(kitti_root: str = "data/raw/kitti"):
    """
    Get KITTI loader, falling back to perturbed KITTI if unavailable.
    """
    if KITTILoader.is_kitti_available(kitti_root):
        return KITTILoader(kitti_root)
    else:
        return FallbackKITTIInterface()
