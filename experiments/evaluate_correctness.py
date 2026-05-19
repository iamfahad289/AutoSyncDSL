"""
Correctness Evaluation

Tests synchronization correctness across different scenarios.
"""

import time
from typing import Dict, List, Tuple
from autosyncdsl import SyncPlan, Executor, Optimizer
from baselines.manual_sync import ManualSyncBaseline
from loaders.kitti_loader import KITTILoader
from loaders.kitti_perturbation import PerturbedScenario


def evaluate_exact_matching() -> Dict:
    """Evaluate exact matching correctness."""
    results = {
        "test": "exact_matching",
        "dsl_correct": False,
        "baseline_correct": False,
        "groups_dsl": 0,
        "groups_baseline": 0,
    }

    # Create stream with exact timestamps
    cam_stream = []
    lidar_stream = []
    imu_stream = []

    for i in range(10):
        ts = i * 10.0
        cam_stream.append(
            __import__("autosyncdsl").SensorReading(
                timestamp_ms=ts, sensor_name="cam",
                sensor_type="camera", data={"id": i}
            )
        )
        lidar_stream.append(
            __import__("autosyncdsl").SensorReading(
                timestamp_ms=ts, sensor_name="lidar",
                sensor_type="lidar", data={"id": i}
            )
        )
        imu_stream.append(
            __import__("autosyncdsl").SensorReading(
                timestamp_ms=ts, sensor_name="imu",
                sensor_type="imu", data={"id": i}
            )
        )

    # DSL approach
    plan = (
        SyncPlan()
          .camera("cam")
          .lidar("lidar")
          .imu("imu")
          .exact("cam")
          .batch(size=4)
    )

    ir = plan.compile()
    executor = Executor(ir)
    dsl_batches = executor.run(cam_stream, lidar_stream, imu_stream)

    # Baseline approach
    baseline = ManualSyncBaseline(
        match_type="exact",
        batch_size=4,
    )
    baseline_batches = baseline.synchronize(cam_stream, lidar_stream, imu_stream)

    # Check correctness
    dsl_group_count = sum(len(batch) for batch in dsl_batches)
    baseline_group_count = sum(len(batch) for batch in baseline_batches)

    results["groups_dsl"] = dsl_group_count
    results["groups_baseline"] = baseline_group_count

    # For exact matching with identical timestamps, should have 10 groups
    results["dsl_correct"] = dsl_group_count == 10
    results["baseline_correct"] = baseline_group_count == 10

    return results


def evaluate_nearest_matching() -> Dict:
    """Evaluate nearest matching correctness."""
    results = {
        "test": "nearest_matching",
        "dsl_correct": False,
        "baseline_correct": False,
        "tolerance_ms": 50.0,
    }

    cam_stream = []
    lidar_stream = []
    imu_stream = []

    # Camera at regular intervals
    for i in range(5):
        cam_stream.append(
            __import__("autosyncdsl").SensorReading(
                timestamp_ms=i * 100.0, sensor_name="cam",
                sensor_type="camera", data={"id": i}
            )
        )

    # LiDAR slightly offset (within tolerance)
    for i in range(5):
        lidar_stream.append(
            __import__("autosyncdsl").SensorReading(
                timestamp_ms=i * 100.0 + 10.0, sensor_name="lidar",
                sensor_type="lidar", data={"id": i}
            )
        )

    # IMU aligned
    for i in range(5):
        imu_stream.append(
            __import__("autosyncdsl").SensorReading(
                timestamp_ms=i * 100.0, sensor_name="imu",
                sensor_type="imu", data={"id": i}
            )
        )

    # DSL approach
    plan = (
        SyncPlan()
          .camera("cam")
          .lidar("lidar")
          .imu("imu")
          .nearest("cam", tolerance_ms=50)
          .batch(size=5)
    )

    ir = plan.compile()
    executor = Executor(ir)
    dsl_batches = executor.run(cam_stream, lidar_stream, imu_stream)

    # Baseline approach
    baseline = ManualSyncBaseline(
        match_type="nearest",
        tolerance_ms=50,
        batch_size=5,
    )
    baseline_batches = baseline.synchronize(cam_stream, lidar_stream, imu_stream)

    # Check that both matched readings
    dsl_has_matches = any(
        len(group.lidar_readings) > 0
        for batch in dsl_batches
        for group in batch
    )

    baseline_has_matches = any(
        len(group.lidar_readings) > 0
        for batch in baseline_batches
        for group in batch
    )

    results["dsl_correct"] = dsl_has_matches
    results["baseline_correct"] = baseline_has_matches

    return results


def evaluate_stale_filtering() -> Dict:
    """Evaluate stale frame filtering."""
    results = {
        "test": "stale_filtering",
        "dsl_kept": 0,
        "dsl_filtered": 0,
        "baseline_kept": 0,
        "baseline_filtered": 0,
    }

    cam_stream = []
    lidar_stream = []
    imu_stream = []

    # Camera stream
    for i in range(3):
        cam_stream.append(
            __import__("autosyncdsl").SensorReading(
                timestamp_ms=i * 100.0, sensor_name="cam",
                sensor_type="camera", data={"id": i}
            )
        )

    # LiDAR: first two within tolerance, last one is stale
    lidar_stream = [
        __import__("autosyncdsl").SensorReading(
            timestamp_ms=0.0, sensor_name="lidar",
            sensor_type="lidar", data={"id": 0}
        ),
        __import__("autosyncdsl").SensorReading(
            timestamp_ms=50.0, sensor_name="lidar",
            sensor_type="lidar", data={"id": 1}
        ),
        __import__("autosyncdsl").SensorReading(
            timestamp_ms=50.0, sensor_name="lidar",
            sensor_type="lidar", data={"id": 2}
        ),  # at 200ms, this lidar at 50ms is 150ms old
    ]

    # IMU aligned
    for i in range(3):
        imu_stream.append(
            __import__("autosyncdsl").SensorReading(
                timestamp_ms=i * 100.0, sensor_name="imu",
                sensor_type="imu", data={"id": i}
            )
        )

    # DSL with stale filtering
    plan = (
        SyncPlan()
          .camera("cam")
          .lidar("lidar")
          .imu("imu")
          .nearest("cam", tolerance_ms=100)
          .drop_stale(max_age_ms=100)
          .batch(size=3)
    )

    ir = plan.compile()
    executor = Executor(ir)
    dsl_batches = executor.run(cam_stream, lidar_stream, imu_stream)
    dsl_count = sum(len(batch) for batch in dsl_batches)

    # Baseline
    baseline = ManualSyncBaseline(
        match_type="nearest",
        tolerance_ms=100,
        max_age_ms=100,
        batch_size=3,
    )
    baseline_batches = baseline.synchronize(cam_stream, lidar_stream, imu_stream)
    baseline_count = sum(len(batch) for batch in baseline_batches)

    results["dsl_kept"] = dsl_count
    results["baseline_kept"] = baseline_count

    return results


def evaluate_kitti_real_drive(max_frames: int = 50) -> Dict:
    """Evaluate correctness on the downloaded real KITTI drive."""
    loader = KITTILoader("data/raw/kitti")
    drives = loader.load_drives()
    if not drives:
        return {"test": "kitti_real", "status": "skipped", "reason": "KITTI drive not available"}

    drive_name = drives[0]
    camera_stream, lidar_stream, imu_stream = loader.load_sequence(drive_name, max_frames=max_frames)

    plan = (
        SyncPlan()
          .camera("camera")
          .lidar("lidar")
          .imu("imu")
          .nearest("camera", tolerance_ms=75)
          .interpolate("imu", window_ms=120)
          .drop_stale(max_age_ms=250)
          .batch(size=4)
    )

    ir = plan.compile()
    executor = Executor(ir)
    dsl_batches = executor.run(camera_stream, lidar_stream, imu_stream)

    baseline = ManualSyncBaseline(
        match_type="nearest",
        tolerance_ms=75,
        max_age_ms=250,
        batch_size=4,
    )
    baseline_batches = baseline.synchronize(camera_stream, lidar_stream, imu_stream)

    def summarize(batches):
        groups = [g for batch in batches for g in batch]
        lidar_offsets = []
        imu_offsets = []
        interpolated = 0
        for g in groups:
            if g.camera_readings and g.lidar_readings:
                lidar_offsets.append(abs(g.camera_readings[0].timestamp_ms - g.lidar_readings[0].timestamp_ms))
            if g.camera_readings and g.imu_readings:
                imu = g.imu_readings[0]
                imu_offsets.append(abs(g.camera_readings[0].timestamp_ms - imu.timestamp_ms))
                if imu.data.get("interpolated"):
                    interpolated += 1
        return {
            "group_count": len(groups),
            "mean_lidar_offset_ms": sum(lidar_offsets) / len(lidar_offsets) if lidar_offsets else 0,
            "mean_imu_offset_ms": sum(imu_offsets) / len(imu_offsets) if imu_offsets else 0,
            "interpolated_groups": interpolated,
        }

    dsl_summary = summarize(dsl_batches)
    baseline_summary = summarize(baseline_batches)

    return {
        "test": "kitti_real",
        "status": "ok",
        "drive": drive_name,
        "frames": {
            "camera": len(camera_stream),
            "lidar": len(lidar_stream),
            "imu": len(imu_stream),
        },
        "dsl": dsl_summary,
        "baseline": baseline_summary,
    }


def run_all_correctness_tests() -> List[Dict]:
    """Run all correctness tests."""
    tests = [
        ("Exact Matching", evaluate_exact_matching),
        ("Nearest Matching", evaluate_nearest_matching),
        ("Stale Filtering", evaluate_stale_filtering),
        ("KITTI Real Drive", evaluate_kitti_real_drive),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            result = test_func()
            results.append(result)
            print(f"  Result: {result}")
        except Exception as e:
            print(f"  Error: {e}")
            results.append({"test": test_name, "error": str(e)})

    return results


if __name__ == "__main__":
    results = run_all_correctness_tests()
    print("\n\n=== Summary ===")
    for r in results:
        print(r)

