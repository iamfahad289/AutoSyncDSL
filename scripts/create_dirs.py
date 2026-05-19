import os

dirs_to_create = [
    "final_submission/results",
    "final_submission/tables",
    "final_submission/figures/png",
    "final_submission/figures/pdf",
    "final_submission/figures/svg",
    "final_submission/figures/drawio",
    "final_submission/logs",
    "final_submission/scripts",
    "final_submission/code_snapshot",
    "final_submission/dataset_summary",
    "final_submission/archive_old_outputs"
]

for d in dirs_to_create:
    os.makedirs(os.path.join("/Users/muhammadfahad/AutoSyncDSL-ClassProject", d), exist_ok=True)
