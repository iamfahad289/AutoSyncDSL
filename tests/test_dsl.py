"""
Unit tests for AutoSyncDSL

Includes tests for:
- DSL compilation
- Exact and nearest matching
- Filtering
- Batching
- Optimizations
"""

import pytest
from autosyncdsl import SyncPlan, Executor, SensorReading, Optimizer
from loaders.kitti_perturbation import PerturbedScenario


class TestDSLCompilation:
    """Test DSL compilation to IR."""

    def test_simple_plan(self):
        """Test compiling a simple DSL plan."""
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .nearest("lidar", tolerance_ms=50)
              .batch(size=4)
        )

        ir = plan.compile()

        assert ir is not None
        assert len(ir.nodes) > 0
        assert len(ir.edges) >= 0
        assert len(ir.sources) == 3

    def test_exact_match_plan(self):
        """Test exact match configuration."""
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .exact("cam")
              .batch(size=4)
        )

        ir = plan.compile()

        # Check for exact match node
        has_exact = any(n.node_type == "exact_match" for n in ir.nodes.values())
        assert has_exact

    def test_ir_serialization(self):
        """Test IR to dict conversion."""
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .nearest("lidar", tolerance_ms=30)
              .batch(size=2)
        )

        ir = plan.compile()
        ir_dict = ir.to_dict()

        assert "nodes" in ir_dict
        assert "edges" in ir_dict
        assert "sources" in ir_dict


class TestExactMatching:
    """Test exact timestamp matching."""

    def test_exact_match_same_timestamps(self):
        """Test exact matching when timestamps are identical."""
        # Create readings with same timestamps
        camera_stream = [
            SensorReading(timestamp_ms=100.0, sensor_name="cam",
                         sensor_type="camera", data={}),
            SensorReading(timestamp_ms=200.0, sensor_name="cam",
                         sensor_type="camera", data={}),
        ]

        lidar_stream = [
            SensorReading(timestamp_ms=100.0, sensor_name="lidar",
                         sensor_type="lidar", data={}),
            SensorReading(timestamp_ms=200.0, sensor_name="lidar",
                         sensor_type="lidar", data={}),
        ]

        imu_stream = [
            SensorReading(timestamp_ms=100.0, sensor_name="imu",
                         sensor_type="imu", data={}),
            SensorReading(timestamp_ms=200.0, sensor_name="imu",
                         sensor_type="imu", data={}),
        ]

        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .exact("cam")
              .batch(size=2)
        )

        ir = plan.compile()
        executor = Executor(ir)
        batches = executor.run(camera_stream, lidar_stream, imu_stream)

        assert len(batches) > 0
        # Each batch should be complete
        for batch in batches:
            for group in batch:
                assert len(group.camera_readings) > 0


class TestNearestMatching:
    """Test nearest-neighbor matching."""

    def test_nearest_match_within_tolerance(self):
        """Test nearest matching within tolerance."""
        camera_stream = [
            SensorReading(timestamp_ms=100.0, sensor_name="cam",
                         sensor_type="camera", data={}),
        ]

        lidar_stream = [
            SensorReading(timestamp_ms=110.0, sensor_name="lidar",
                         sensor_type="lidar", data={}),
        ]

        imu_stream = [
            SensorReading(timestamp_ms=100.5, sensor_name="imu",
                         sensor_type="imu", data={}),
        ]

        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .nearest("cam", tolerance_ms=50)
              .batch(size=1)
        )

        ir = plan.compile()
        executor = Executor(ir)
        batches = executor.run(camera_stream, lidar_stream, imu_stream)

        assert len(batches) > 0
        assert len(batches[0]) > 0

    def test_nearest_match_outside_tolerance(self):
        """Test that readings outside tolerance are not matched."""
        camera_stream = [
            SensorReading(timestamp_ms=100.0, sensor_name="cam",
                         sensor_type="camera", data={}),
        ]

        lidar_stream = [
            SensorReading(timestamp_ms=200.0, sensor_name="lidar",
                         sensor_type="lidar", data={}),
        ]

        imu_stream = [
            SensorReading(timestamp_ms=100.0, sensor_name="imu",
                         sensor_type="imu", data={}),
        ]

        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .nearest("cam", tolerance_ms=50)  # Only 50 ms tolerance
              .batch(size=1)
        )

        ir = plan.compile()
        executor = Executor(ir)
        batches = executor.run(camera_stream, lidar_stream, imu_stream)

        # LiDAR is 100 ms away, should not match
        if len(batches) > 0 and len(batches[0]) > 0:
            group = batches[0][0]
            assert len(group.lidar_readings) == 0


class TestBatching:
    """Test batch construction."""

    def test_batch_size_4(self):
        """Test batches of size 4."""
        cam_stream = [
            SensorReading(timestamp_ms=float(i*10), sensor_name="cam",
                         sensor_type="camera", data={})
            for i in range(10)
        ]

        lidar_stream = [
            SensorReading(timestamp_ms=float(i*10), sensor_name="lidar",
                         sensor_type="lidar", data={})
            for i in range(10)
        ]

        imu_stream = [
            SensorReading(timestamp_ms=float(i*10), sensor_name="imu",
                         sensor_type="imu", data={})
            for i in range(10)
        ]

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
        batches = executor.run(cam_stream, lidar_stream, imu_stream)

        # Should have 3 batches (4 + 4 + 2)
        assert len(batches) == 3
        assert len(batches[0]) == 4
        assert len(batches[1]) == 4
        assert len(batches[2]) == 2


class TestStaleFiltering:
    """Test stale frame rejection."""

    def test_drop_stale_simple(self):
        """Test stale filtering with simple case."""
        # Create a group with old LiDAR reading
        camera_stream = [
            SensorReading(timestamp_ms=200.0, sensor_name="cam",
                         sensor_type="camera", data={}),
        ]

        lidar_stream = [
            SensorReading(timestamp_ms=50.0, sensor_name="lidar",
                         sensor_type="lidar", data={}),
        ]

        imu_stream = [
            SensorReading(timestamp_ms=200.0, sensor_name="imu",
                         sensor_type="imu", data={}),
        ]

        # Camera at 200, LiDAR at 50 (150 ms old)
        # max_age_ms=100, so should be filtered out
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .exact("cam")
              .drop_stale(max_age_ms=100)
              .batch(size=1)
        )

        ir = plan.compile()
        executor = Executor(ir)
        batches = executor.run(camera_stream, lidar_stream, imu_stream)

        # Group should be filtered out since LiDAR is too old
        assert len(batches) == 0


class TestOptimizations:
    """Test optimization passes."""

    def test_rule_fusion_pass(self):
        """Test rule fusion optimization."""
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .imu("imu")
              .nearest("cam", tolerance_ms=50)
              .interpolate("imu", window_ms=20)
              .drop_stale(max_age_ms=100)
              .batch(size=4)
        )

        ir = plan.compile()

        optimizer = Optimizer()
        optimized_ir = optimizer.optimize(ir)

        # Check that optimization was applied
        assert hasattr(optimized_ir, '_optimization_pass_count')
        assert optimized_ir._optimization_pass_count > 0

    def test_buffer_reuse_pass(self):
        """Test buffer reuse optimization."""
        plan = (
            SyncPlan()
              .camera("cam")
              .lidar("lidar")
              .nearest("cam", tolerance_ms=50)
              .interpolate("imu", window_ms=20)
              .batch(size=4)
        )

        ir = plan.compile()

        optimizer = Optimizer()
        optimized_ir = optimizer.optimize(ir)

        # Check for buffer reuse candidates
        assert hasattr(optimized_ir, '_buffer_reuse_candidates')


class TestPerturbedKITTIData:
    """Test perturbed KITTI data generation."""

    def test_clean_scenario(self):
        """Test clean scenario generation."""
        cam, lidar, imu = PerturbedScenario.clean_scenario()

        assert len(cam) > 0
        assert len(lidar) > 0
        assert len(imu) > 0

    def test_high_jitter_scenario(self):
        """Test high jitter scenario."""
        cam, lidar, imu = PerturbedScenario.high_jitter_scenario()

        assert len(cam) > 0
        assert len(lidar) > 0
        assert len(imu) > 0

    def test_dropout_scenario(self):
        """Test dropout scenario."""
        cam, lidar, imu = PerturbedScenario.dropout_scenario()

        # Should have fewer samples due to dropout
        assert len(cam) > 0
        assert len(lidar) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
