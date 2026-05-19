#!/usr/bin/env python3
"""
Run all experiments

Executes correctness, latency, and robustness evaluations.
"""

import sys
import json
import os
from experiments.evaluate_correctness import run_all_correctness_tests
from experiments.evaluate_latency import run_all_latency_tests, print_latency_results
from experiments.evaluate_robustness import run_robustness_tests, print_robustness_results


def main():
    """Run all experiments."""
    os.makedirs("results", exist_ok=True)

    print("="*60)
    print("AUTOSYNCDSL EVALUATION SUITE")
    print("="*60)

    # Correctness tests
    print("\n\n[1] RUNNING CORRECTNESS TESTS\n")
    correctness_results = run_all_correctness_tests()

    # Latency tests
    print("\n\n[2] RUNNING LATENCY TESTS\n")
    latency_results = run_all_latency_tests()
    print_latency_results(latency_results)

    # Robustness tests
    print("\n\n[3] RUNNING ROBUSTNESS TESTS\n")
    robustness_results = run_robustness_tests()
    print_robustness_results(robustness_results)

    # Save results
    print("\n\n[4] SAVING RESULTS\n")

    all_results = {
        "correctness": correctness_results,
        "latency": latency_results,
        "robustness": robustness_results,
    }

    with open("results/all_experiments.json", "w") as f:
        # Convert to serializable format
        serializable = {
            "correctness": correctness_results,
            "latency": latency_results,
            "robustness": robustness_results,
        }
        json.dump(serializable, f, indent=2, default=str)

    print("Results saved to results/all_experiments.json")

    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

