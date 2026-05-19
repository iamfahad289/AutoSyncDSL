"""
AutoSyncDSL - Embedded Python DSL for Multi-Sensor Timestamp Alignment

Main API exports.
"""

from autosyncdsl.dsl import SyncPlan
from autosyncdsl.ir import IR
from autosyncdsl.executor import Executor, SensorReading, SyncGroup
from autosyncdsl.optimizations import Optimizer

__version__ = "0.1.0"
__all__ = [
    "SyncPlan",
    "IR",
    "Executor",
    "SensorReading",
    "SyncGroup",
    "Optimizer",
]

