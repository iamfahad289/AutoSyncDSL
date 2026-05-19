# AutoSyncDSL: Embedded Python DSL for Multi-Sensor Timestamp Alignment

**AutoSyncDSL** is a narrow, focused embedded Python DSL for expressing multi-sensor timestamp alignment and batching policies in autonomous vehicle perception pipelines. This project demonstrates compiler-style lowering, intermediate representation design, and domain-specific optimizations for a practical synchronization problem.

## What is AutoSyncDSL?

In autonomous vehicle perception, multiple sensors (camera, LiDAR, IMU) produce timestamped readings at different rates and with varying latencies. Aligning these streams correctly is critical but tedious to implement by hand. AutoSyncDSL provides:

- **Fluent embedded DSL API** for declaring synchronization policies
- **Small intermediate representation (IR)** representing alignment, filtering, and batching plans
- **Lightweight runtime executor** that processes sensor streams
- **Domain-specific optimizations** including rule fusion and buffer reuse
- **perturbed KITTI data generation** with configurable jitter, drops, and delays
- **Evaluation framework** comparing DSL approach against handwritten baseline

## Key Features

- **Exact and nearest-neighbor timestamp matching** with configurable tolerances
- **IMU interpolation** between adjacent timestamps
- **Stale-frame rejection** based on maximum age constraints
- **Fixed-size batch construction** for efficient processing
- **Rule fusion optimization** combining compatible alignment steps
- **Buffer reuse optimization** for reduced allocations
- **Comprehensive evaluation** on correctness, latency, and code complexity

## Project Structure

```
autosyncdsl/
├── autosyncdsl/           # Core DSL and runtime
│   ├── dsl.py            # Fluent DSL API
│   ├── ir.py             # Intermediate representation
│   ├── executor.py       # Runtime executor
│   ├── optimizations.py  # Optimization passes
│   ├── policies.py       # Matching and filtering policies
│   ├── batching.py       # Batch construction
│   └── utils.py          # Utility functions
├── loaders/              # Data loading
│   ├── perturbed KITTI_loader.py
│   └── kitti_loader.py
├── baselines/            # Handwritten baseline
│   └── manual_sync.py
├── experiments/          # Evaluation scripts
├── tests/                # Unit tests
├── scripts/              # Utility scripts
├── data/                 # Data directory
└── docs/                 # Documentation and proposals
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate perturbed KITTI Data

```bash
python scripts/generate_perturbed KITTI_sync_data.py
```

### 3. Run Experiments

```bash
python scripts/run_all_experiments.py
```

### 4. View Results

```bash
python scripts/export_figures.py
```

## DSL Example Usage

```python
from autosyncdsl import SyncPlan

# Define a synchronization policy
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

# Compile to IR
ir = plan.compile()

# Execute on sensor streams
executor = Executor(ir)
batches = executor.run(camera_stream, lidar_stream, imu_stream)

for batch in batches:
    print(f"Camera: {batch.camera}, LiDAR: {batch.lidar}, IMU: {len(batch.imu)} samples")
```

## Key Components

### DSL (`dsl.py`)

Fluent Python API for expressing synchronization policies. Returns a plan that can be compiled to IR.

### Intermediate Representation (`ir.py`)

Small, inspectable representation of the synchronization plan:
- Sensor source nodes
- Match policy nodes (exact, nearest)
- Filter nodes (stale rejection)
- Interpolation nodes (IMU)
- Batch nodes

### Executor (`executor.py`)

Runtime that processes sensor streams according to IR and produces synchronized batches. Includes:
- Stream buffering and ordering
- Timestamp matching
- Interpolation logic
- Batch assembly

### Optimizations (`optimizations.py`)

1. **Rule Fusion**: Combines compatible alignment and filtering nodes
2. **Buffer Reuse**: Reuses temporary structures to reduce allocations

## Experimental Evaluation

The project includes comprehensive evaluation:

### Correctness
- Exact-match behavior validation
- Nearest-match tolerance verification
- Interpolation accuracy
- Batch composition correctness

### Latency
- Batch construction time
- Memory overhead
- Comparison with baseline

### Robustness
- Low-jitter scenario
- Medium-jitter scenario
- Frame drop scenarios
- Different tolerance windows

### Code Complexity
- Lines of code (DSL approach vs baseline)
- Implementation complexity metrics

## Dataset

### perturbed KITTI Data
Automatically generated with:
- Configurable sensor rates (e.g., camera 30 Hz, LiDAR 10 Hz, IMU 100 Hz)
- Realistic jitter (0-50 ms range)
- Dropped frames (0-5% drop rate)
- Out-of-order deliveries
- Configurable tolerance windows

### KITTI Dataset (Optional)
The project attempts to load a small KITTI Raw subset if available:

```bash
python scripts/prepare_kitti_subset.py
```

If KITTI download is unavailable, the project runs fully on perturbed KITTI data. Instructions for manually adding KITTI data are in `KITTI_SETUP.md`.

## Current Limitations

1. **Scope**: Only three sensor types (camera, LiDAR, IMU)
2. **Stream Processing**: Batch-oriented, not true streaming
3. **Optimization Scope**: Two lightweight passes (fusion and buffer reuse)
4. **Dataset**: perturbed KITTI primary; optional KITTI as real-data fallback
5. **No Model Training**: Pure systems/DSL project, not ML-based

## Next Steps for Full Project

- Extend to additional sensor types (radar, event cameras)
- Implement advanced graph-based optimization passes
- Add real-time streaming support
- Integration with ROS 2 message_filters
- Extended KITTI support with automatic download
- Performance profiling and benchmarking

## References

See `docs/references.bib` for academic citations related to ROS, MLIR, TVM, and KITTI.

## License

MIT

## Contact

For questions or contributions, please open an issue or pull request.

