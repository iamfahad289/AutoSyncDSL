"""
Robustness Evaluation

Tests system behavior under various stress conditions.
"""

from typing import Dict, List
from autosyncdsl import SyncPlan, Executor
from baselines.manual_sync import ManualSyncBaseline
from loaders.kitti_perturbation import KITTIPerturbationGenerator, PerturbedScenario


def evaluate_various_jitter_levels() -> List[Dict]:
    """Evaluate under various jitter levels."""
    results = []
    jitter_levels = [2.0, 5.0, 10.0, 20.0, 30.0, 50.0]

    for jitter_ms in jitter_levels:
        print(f"  Testing jitter={jitter_ms}ms...")

        gen = KITTIPerturbationGenerator(
            duration_sec=5.0,
            random_seed=42,
        )
        cam, lidar, imu = gen.generate(jitter_ms=jitter_ms, dropout_rate=0.0)

        # DSL approach
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .nearest("lidar", tolerance_ms=50)
              .batch(size=4)
        )

        ir = plan.compile()
        executor = Executor(ir)
        dsl_batches = executor.run(cam, lidar, imu)
        dsl_groups = sum(len(b) for b in dsl_batches)

        # Baseline
        baseline = ManualSyncBaseline(
            match_type="nearest",
            tolerance_ms=50,
            batch_size=4,
        )
        baseline_batches = baseline.synchronize(cam, lidar, imu)
        baseline_groups = sum(len(b) for b in baseline_batches)

        results.append({
            "jitter_ms": jitter_ms,
            "camera_frames": len(cam),
            "lidar_frames": len(lidar),
            "imu_frames": len(imu),
            "dsl_groups": dsl_groups,
            "baseline_groups": baseline_groups,
            "agreement": dsl_groups == baseline_groups,
        })

    return results


def evaluate_various_dropout_rates() -> List[Dict]:
    """Evaluate under various dropout rates."""
    results = []
    dropout_rates = [0.0, 0.01, 0.02, 0.05, 0.10]

    for dropout_rate in dropout_rates:
        print(f"  Testing dropout={dropout_rate*100:.1f}%...")

        gen = KITTIPerturbationGenerator(
            duration_sec=5.0,
            random_seed=42,
        )
        cam, lidar, imu = gen.generate(jitter_ms=5.0, dropout_rate=dropout_rate)

        # DSL approach
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .nearest("lidar", tolerance_ms=50)
              .batch(size=4)
        )

        ir = plan.compile()
        executor = Executor(ir)
        dsl_batches = executor.run(cam, lidar, imu)
        dsl_groups = sum(len(b) for b in dsl_batches)

        # Baseline
        baseline = ManualSyncBaseline(
            match_type="nearest",
            tolerance_ms=50,
            batch_size=4,
        )
        baseline_batches = baseline.synchronize(cam, lidar, imu)
        baseline_groups = sum(len(b) for b in baseline_batches)

        results.append({
            "dropout_rate": dropout_rate,
            "camera_frames": len(cam),
            "lidar_frames": len(lidar),
            "imu_frames": len(imu),
            "dsl_groups": dsl_groups,
            "baseline_groups": baseline_groups,
            "agreement": dsl_groups == baseline_groups,
        })

    return results


def evaluate_various_tolerances() -> List[Dict]:
    """Evaluate with various tolerance windows."""
    results = []
    tolerances = [10.0, 25.0, 50.0, 100.0, 200.0]

    gen = KITTIPerturbationGenerator(
        duration_sec=5.0,
        random_seed=42,
    )
    cam, lidar, imu = gen.generate(jitter_ms=20.0, dropout_rate=0.01)

    for tolerance_ms in tolerances:
        print(f"  Testing tolerance={tolerance_ms}ms...")

        # DSL approach
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .nearest("lidar", tolerance_ms=tolerance_ms)
              .batch(size=4)
        )

        ir = plan.compile()
        executor = Executor(ir)
        dsl_batches = executor.run(cam, lidar, imu)
        dsl_groups = sum(len(b) for b in dsl_batches)

        # Baseline
        baseline = ManualSyncBaseline(
            match_type="nearest",
            tolerance_ms=tolerance_ms,
            batch_size=4,
        )
        baseline_batches = baseline.synchronize(cam, lidar, imu)
        baseline_groups = sum(len(b) for b in baseline_batches)

        results.append({
            "tolerance_ms": tolerance_ms,
            "dsl_groups": dsl_groups,
            "baseline_groups": baseline_groups,
            "agreement": dsl_groups == baseline_groups,
        })

    return results


def run_robustness_tests() -> Dict:
    """Run all robustness tests."""
    print("\nEvaluating various jitter levels...")
    jitter_results = evaluate_various_jitter_levels()

    print("\nEvaluating various dropout rates...")
    dropout_results = evaluate_various_dropout_rates()

    print("\nEvaluating various tolerance windows...")
    tolerance_results = evaluate_various_tolerances()

    return {
        "jitter": jitter_results,
        "dropout": dropout_results,
        "tolerance": tolerance_results,
    }


def print_robustness_results(results: Dict) -> None:
    """Pretty-print robustness results."""
    print("\n=== JITTER ROBUSTNESS ===")
    print(f"{'Jitter':<10} {'Cam':<6} {'Lidar':<6} {'DSL':<6} {'Base':<6} {'Agree':<6}")
    print("-" * 50)
    for r in results["jitter"]:
        agree = "✓" if r["agreement"] else "✗"
        print(f"{r['jitter_ms']:<10.1f} {r['camera_frames']:<6} {r['lidar_frames']:<6} "
              f"{r['dsl_groups']:<6} {r['baseline_groups']:<6} {agree:<6}")

    print("\n=== DROPOUT ROBUSTNESS ===")
    print(f"{'Dropout':<10} {'Cam':<6} {'Lidar':<6} {'DSL':<6} {'Base':<6} {'Agree':<6}")
    print("-" * 50)
    for r in results["dropout"]:
        agree = "✓" if r["agreement"] else "✗"
        print(f"{r['dropout_rate']:<10.2%} {r['camera_frames']:<6} {r['lidar_frames']:<6} "
              f"{r['dsl_groups']:<6} {r['baseline_groups']:<6} {agree:<6}")

    print("\n=== TOLERANCE ROBUSTNESS ===")
    print(f"{'Tolerance':<12} {'DSL Groups':<12} {'Base Groups':<12} {'Agree':<6}")
    print("-" * 42)
    for r in results["tolerance"]:
        agree = "✓" if r["agreement"] else "✗"
        print(f"{r['tolerance_ms']:<12.1f} {r['dsl_groups']:<12} {r['baseline_groups']:<12} {agree:<6}")


if __name__ == "__main__":
    results = run_robustness_tests()
    print_robustness_results(results)

