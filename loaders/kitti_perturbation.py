"""
perturbed KITTI Data Generator for AutoSyncDSL

Generates multi-sensor stream data with configurable:
- Sensor rates
- Jitter/latency
- Dropped frames
- Out-of-order deliveries
"""

import random
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import os

from autosyncdsl.executor import SensorReading
from loaders.kitti_loader import KITTILoader


class KITTIPerturbationGenerator:
    """
    Generate stream data by loading REAL KITTI data and applying optional variations
    like jitter, dropped frames, or out-of-order deliveries.
    (Repurposed to use ONLY real dataset as per user constraints).
    """

    def __init__(
        self,
        duration_sec: float = 10.0,
        camera_rate_hz: float = 30.0,
        lidar_rate_hz: float = 10.0,
        imu_rate_hz: float = 100.0,
        random_seed: int = 42,
    ):
        """
        Initialize generator. Rates and duration are somewhat ignored as we load
        the real KITTI sequence directly.
        """
        self.duration_sec = duration_sec
        self.random_seed = random_seed

        random.seed(random_seed)
        np.random.seed(random_seed)

        self.loader = KITTILoader("data/raw/kitti")

    def generate(
        self,
        jitter_ms: float = 5.0,
        dropout_rate: float = 0.0,
        out_of_order_rate: float = 0.0,
    ) -> Tuple[List[SensorReading], List[SensorReading], List[SensorReading]]:
        """
        Load REAL KITTI sequence and then apply the variations.
        """
        drives = self.loader.load_drives()
        if not drives:
            raise FileNotFoundError("KITTI drive not found. Please ensure real data is available.")

        # Load up to enough frames to cover duration_sec (~10 frames/sec for KITTI lidar)
        max_frames = int(max(self.duration_sec * 30, 200))
        cam, lidar, imu = self.loader.load_sequence(drives[0], max_frames=max_frames)

        cam = self._apply_variations(cam, jitter_ms, dropout_rate, out_of_order_rate)
        lidar = self._apply_variations(lidar, jitter_ms, dropout_rate, out_of_order_rate)
        imu = self._apply_variations(imu, jitter_ms, dropout_rate, out_of_order_rate)

        return cam, lidar, imu

    def _apply_variations(
        self,
        stream: List[SensorReading],
        jitter_ms: float,
        dropout_rate: float,
        out_of_order_rate: float,
    ) -> List[SensorReading]:
        """Apply noise to a real stream."""
        if not stream:
            return stream

        result = []
        for r in stream:
            if random.random() < dropout_rate:
                continue

            jitter = np.random.normal(0, jitter_ms)
            r.timestamp_ms += jitter
            result.append(r)

        if out_of_order_rate > 0:
            for i in range(len(result) - 1):
                if random.random() < out_of_order_rate:
                    result[i], result[i + 1] = result[i + 1], result[i]

        return result

class PerturbedScenario:
    """
    Pre-defined scenarios for testing different conditions.
    """

    @staticmethod
    def clean_scenario() -> Tuple[
        List[SensorReading], List[SensorReading], List[SensorReading]
    ]:
        """Clean synchronized scenario with minimal jitter."""
        gen = KITTIPerturbationGenerator(duration_sec=5.0)
        return gen.generate(jitter_ms=2.0, dropout_rate=0.0)

    @staticmethod
    def moderate_jitter_scenario() -> Tuple[
        List[SensorReading], List[SensorReading], List[SensorReading]
    ]:
        """Scenario with moderate jitter."""
        gen = KITTIPerturbationGenerator(duration_sec=5.0)
        return gen.generate(jitter_ms=10.0, dropout_rate=0.01)

    @staticmethod
    def high_jitter_scenario() -> Tuple[
        List[SensorReading], List[SensorReading], List[SensorReading]
    ]:
        """Scenario with high jitter."""
        gen = KITTIPerturbationGenerator(duration_sec=5.0)
        return gen.generate(jitter_ms=30.0, dropout_rate=0.02)

    @staticmethod
    def dropout_scenario() -> Tuple[
        List[SensorReading], List[SensorReading], List[SensorReading]
    ]:
        """Scenario with significant frame drops."""
        gen = KITTIPerturbationGenerator(duration_sec=5.0)
        return gen.generate(jitter_ms=5.0, dropout_rate=0.05)

    @staticmethod
    def extreme_scenario() -> Tuple[
        List[SensorReading], List[SensorReading], List[SensorReading]
    ]:
        """Extreme scenario with high jitter, drops, and disorder."""
        gen = KITTIPerturbationGenerator(duration_sec=5.0)
        return gen.generate(
            jitter_ms=50.0,
            dropout_rate=0.05,
            out_of_order_rate=0.02,
        )


def save_scenario(
    camera_stream: List[SensorReading],
    lidar_stream: List[SensorReading],
    imu_stream: List[SensorReading],
    filepath: str,
) -> None:
    """Save scenario to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    data = {
        "camera": [
            {
                "timestamp_ms": r.timestamp_ms,
                "raw_timestamp": r.data.get("raw_timestamp"),
            }
            for r in camera_stream
        ],
        "lidar": [
            {
                "timestamp_ms": r.timestamp_ms,
                "raw_timestamp": r.data.get("raw_timestamp"),
            }
            for r in lidar_stream
        ],
        "imu": [
            {
                "timestamp_ms": r.timestamp_ms,
                "raw_timestamp": r.data.get("raw_timestamp"),
            }
            for r in imu_stream
        ],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_scenario(filepath: str) -> Tuple[
    List[SensorReading], List[SensorReading], List[SensorReading]
]:
    """Load scenario from file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    camera_stream = [
        SensorReading(
            timestamp_ms=r["timestamp_ms"],
            sensor_name="camera",
            sensor_type="camera",
            data={"raw_timestamp": r["raw_timestamp"]},
        )
        for r in data["camera"]
    ]

    lidar_stream = [
        SensorReading(
            timestamp_ms=r["timestamp_ms"],
            sensor_name="lidar",
            sensor_type="lidar",
            data={"raw_timestamp": r["raw_timestamp"]},
        )
        for r in data["lidar"]
    ]

    imu_stream = [
        SensorReading(
            timestamp_ms=r["timestamp_ms"],
            sensor_name="imu",
            sensor_type="imu",
            data={"raw_timestamp": r["raw_timestamp"]},
        )
        for r in data["imu"]
    ]

    return camera_stream, lidar_stream, imu_stream

