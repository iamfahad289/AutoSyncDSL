#!/usr/bin/env python3
"""
Generate perturbed KITTI synchronization data for testing

Creates multiple scenarios with varying jitter, dropout, and complexity.
"""

import os
import sys
from loaders.kitti_perturbation import (
    KITTIPerturbationGenerator, PerturbedScenario, save_scenario
)


def main():
    """Generate all perturbed KITTI scenarios."""
    os.makedirs("data/perturbed_kitti", exist_ok=True)

    print("Generating perturbed KITTI data scenarios...\n")

    # Scenario 1: Clean
    print("1. Clean scenario (low jitter, no dropout)...")
    cam, lidar, imu = PerturbedScenario.clean_scenario()
    save_scenario(cam, lidar, imu, "data/perturbed_kitti/clean_scenario.json")
    print(f"   Saved: camera={len(cam)}, lidar={len(lidar)}, imu={len(imu)}")

    # Scenario 2: Moderate jitter
    print("2. Moderate jitter scenario...")
    cam, lidar, imu = PerturbedScenario.moderate_jitter_scenario()
    save_scenario(cam, lidar, imu, "data/perturbed_kitti/moderate_jitter_scenario.json")
    print(f"   Saved: camera={len(cam)}, lidar={len(lidar)}, imu={len(imu)}")

    # Scenario 3: High jitter
    print("3. High jitter scenario...")
    cam, lidar, imu = PerturbedScenario.high_jitter_scenario()
    save_scenario(cam, lidar, imu, "data/perturbed_kitti/high_jitter_scenario.json")
    print(f"   Saved: camera={len(cam)}, lidar={len(lidar)}, imu={len(imu)}")

    # Scenario 4: Dropout
    print("4. Dropout scenario...")
    cam, lidar, imu = PerturbedScenario.dropout_scenario()
    save_scenario(cam, lidar, imu, "data/perturbed_kitti/dropout_scenario.json")
    print(f"   Saved: camera={len(cam)}, lidar={len(lidar)}, imu={len(imu)}")

    # Scenario 5: Extreme
    print("5. Extreme scenario...")
    cam, lidar, imu = PerturbedScenario.extreme_scenario()
    save_scenario(cam, lidar, imu, "data/perturbed_kitti/extreme_scenario.json")
    print(f"   Saved: camera={len(cam)}, lidar={len(lidar)}, imu={len(imu)}")

    # Scenario 6: Long stream
    print("6. Long stream (30 seconds)...")
    gen = KITTIPerturbationGenerator(
        duration_sec=30.0,
        camera_rate_hz=30,
        lidar_rate_hz=10,
        imu_rate_hz=100,
    )
    cam, lidar, imu = gen.generate(jitter_ms=10.0, dropout_rate=0.02)
    save_scenario(cam, lidar, imu, "data/perturbed_kitti/long_stream_scenario.json")
    print(f"   Saved: camera={len(cam)}, lidar={len(lidar)}, imu={len(imu)}")

    print("\nDone! Scenarios saved to data/perturbed_kitti/")


if __name__ == "__main__":
    main()

