"""
Latency Evaluation

Compares execution latency between DSL approach and baselines.
"""

import time
from typing import Dict, List, Tuple
import statistics

from autosyncdsl import SyncPlan, Executor, Optimizer
from baselines.manual_sync import ManualSyncBaseline, ManualSyncVariant2, SimpleSync
from loaders.kitti_loader import KITTILoader
from loaders.kitti_perturbation import PerturbedScenario, KITTIPerturbationGenerator


def measure_dsl_execution(
    camera_stream, lidar_stream, imu_stream, num_runs: int = 5
) -> Dict:
    """Measure DSL execution time."""
    times = []

    for _ in range(num_runs):
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .nearest("lidar", tolerance_ms=50)
              .drop_stale(max_age_ms=100)
              .batch(size=4)
        )

        ir = plan.compile()

        # Measure execution only, not compilation
        start = time.perf_counter()
        executor = Executor(ir)
        batches = executor.run(camera_stream, lidar_stream, imu_stream)
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

    return {
        "approach": "DSL",
        "times_ms": times,
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def measure_dsl_with_optimization(
    camera_stream, lidar_stream, imu_stream, num_runs: int = 5
) -> Dict:
    """Measure DSL execution with optimization."""
    times = []

    for _ in range(num_runs):
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .nearest("lidar", tolerance_ms=50)
              .drop_stale(max_age_ms=100)
              .batch(size=4)
        )

        ir = plan.compile()
        optimizer = Optimizer()
        optimized_ir = optimizer.optimize(ir)

        start = time.perf_counter()
        executor = Executor(optimized_ir)
        batches = executor.run(camera_stream, lidar_stream, imu_stream)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return {
        "approach": "DSL+Opt",
        "times_ms": times,
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def measure_baseline_v1(
    camera_stream, lidar_stream, imu_stream, num_runs: int = 5
) -> Dict:
    """Measure baseline v1 execution time."""
    times = []

    for _ in range(num_runs):
        baseline = ManualSyncBaseline(
            match_type="nearest",
            tolerance_ms=50,
            max_age_ms=100,
            batch_size=4,
        )

        start = time.perf_counter()
        batches = baseline.synchronize(camera_stream, lidar_stream, imu_stream)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return {
        "approach": "Baseline-v1",
        "times_ms": times,
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def measure_baseline_v2(
    camera_stream, lidar_stream, imu_stream, num_runs: int = 5
) -> Dict:
    """Measure baseline v2 execution time."""
    times = []

    for _ in range(num_runs):
        baseline = ManualSyncVariant2(
            tolerance_ms=50,
            max_age_ms=100,
            batch_size=4,
        )

        start = time.perf_counter()
        batches = baseline.synchronize(camera_stream, lidar_stream, imu_stream)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return {
        "approach": "Baseline-v2",
        "times_ms": times,
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def measure_simple_sync(
    camera_stream, lidar_stream, imu_stream, num_runs: int = 5
) -> Dict:
    """Measure simple sync execution time."""
    times = []

    for _ in range(num_runs):
        baseline = SimpleSync(window_ms=100, batch_size=4)

        start = time.perf_counter()
        batches = baseline.synchronize(camera_stream, lidar_stream, imu_stream)
        end = time.perf_counter()

        times.append((end - start) * 1000)

    return {
        "approach": "SimpleSync",
        "times_ms": times,
        "mean_ms": statistics.mean(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def evaluate_latency_clean() -> List[Dict]:
    """Evaluate latency on clean scenario."""
    print("Generating clean scenario...")
    cam, lidar, imu = PerturbedScenario.clean_scenario()

    print(f"  Camera: {len(cam)} frames")
    print(f"  LiDAR: {len(lidar)} frames")
    print(f"  IMU: {len(imu)} frames")

    results = []

    print("Measuring DSL...")
    results.append(measure_dsl_execution(cam, lidar, imu, num_runs=5))

    print("Measuring DSL+Opt...")
    results.append(measure_dsl_with_optimization(cam, lidar, imu, num_runs=5))

    print("Measuring Baseline-v1...")
    results.append(measure_baseline_v1(cam, lidar, imu, num_runs=5))

    print("Measuring Baseline-v2...")
    results.append(measure_baseline_v2(cam, lidar, imu, num_runs=5))

    print("Measuring SimpleSync...")
    results.append(measure_simple_sync(cam, lidar, imu, num_runs=5))

    return results


def evaluate_latency_jittery() -> List[Dict]:
    """Evaluate latency on jittery scenario."""
    print("\nGenerating jittery scenario...")
    cam, lidar, imu = PerturbedScenario.high_jitter_scenario()

    print(f"  Camera: {len(cam)} frames")
    print(f"  LiDAR: {len(lidar)} frames")
    print(f"  IMU: {len(imu)} frames")

    results = []

    print("Measuring DSL...")
    results.append(measure_dsl_execution(cam, lidar, imu, num_runs=5))

    print("Measuring Baseline-v1...")
    results.append(measure_baseline_v1(cam, lidar, imu, num_runs=5))

    return results


def evaluate_latency_large_stream() -> List[Dict]:
    """Evaluate latency on larger stream."""
    print("\nGenerating large stream...")
    gen = KITTIPerturbationGenerator(
        duration_sec=30.0,
        camera_rate_hz=30,
        lidar_rate_hz=10,
        imu_rate_hz=100,
    )
    cam, lidar, imu = gen.generate(jitter_ms=10)

    print(f"  Camera: {len(cam)} frames")
    print(f"  LiDAR: {len(lidar)} frames")
    print(f"  IMU: {len(imu)} frames")

    results = []

    print("Measuring DSL...")
    results.append(measure_dsl_execution(cam, lidar, imu, num_runs=3))

    print("Measuring Baseline-v1...")
    results.append(measure_baseline_v1(cam, lidar, imu, num_runs=3))

    return results


def evaluate_latency_kitti_real_drive(num_runs: int = 5, max_frames: int = 50) -> List[Dict]:
    """Evaluate latency on the downloaded real KITTI drive."""
    loader = KITTILoader("data/raw/kitti")
    drives = loader.load_drives()
    if not drives:
        return [{"approach": "KITTI-Real", "status": "skipped", "reason": "KITTI drive not available"}]

    drive_name = drives[0]
    cam, lidar, imu = loader.load_sequence(drive_name, max_frames=max_frames)

    results = []

    results.append(measure_dsl_execution(cam, lidar, imu, num_runs=num_runs))
    results.append(measure_dsl_with_optimization(cam, lidar, imu, num_runs=num_runs))
    results.append(measure_baseline_v1(cam, lidar, imu, num_runs=num_runs))
    results.append(measure_baseline_v2(cam, lidar, imu, num_runs=num_runs))
    results.append(measure_simple_sync(cam, lidar, imu, num_runs=num_runs))

    for r in results:
        r["scenario"] = "kitti_real"
        r["drive"] = drive_name
        r["frames"] = {"camera": len(cam), "lidar": len(lidar), "imu": len(imu)}

    return results


def run_all_latency_tests() -> Dict:
    """Run all latency tests."""
    all_results = {
        "clean": evaluate_latency_clean(),
        "jittery": evaluate_latency_jittery(),
        "large": evaluate_latency_large_stream(),
        "kitti_real": evaluate_latency_kitti_real_drive(),
    }

    return all_results


def print_latency_results(all_results: Dict) -> None:
    """Pretty-print latency results."""
    for scenario, results in all_results.items():
        print(f"\n=== {scenario.upper()} SCENARIO ===")
        for r in results:
            print(f"\n{r['approach']}:")
            print(f"  Mean: {r['mean_ms']:.4f} ms")
            print(f"  StDev: {r['stdev_ms']:.4f} ms")
            print(f"  Min: {r['min_ms']:.4f} ms")
            print(f"  Max: {r['max_ms']:.4f} ms")


if __name__ == "__main__":
    all_results = run_all_latency_tests()
    print_latency_results(all_results)

