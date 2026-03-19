"""
Lightweight text + JSON logging for a single cbspec pipeline run.

This module provides the RunLogger class, which writes:
    - A human-readable text log:
        output/runs/<timestamp>/logs/runs.log

    - A machine-readable JSONL log:
        output/runs/<timestamp>/logs/runs.jsonl
"""

import json
from datetime import datetime
from pathlib import Path

class RunLogger:
    """
    Handles both text and JSON output and logging for a single pipeline run.

    :param logs_dir: Path
                     Directory where log files will be written. This directory is
                     created automatically if it does not already exist.

    Notes:
        - Two files are opened:
            run.log     → human-readable text log
            run.jsonl   → structured JSON log (one entry per line)

        - Files are opened in append mode so multiple pipeline stages can write
          to the same log without overwriting previous entries.

        - The logger must be closed at the end of the run to flush file handles.
    """

    def __init__(self, logs_dir: Path):
        # Ensure directory exists
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.text_path = self.logs_dir / "run.log"
        self.json_path = self.logs_dir / "run.jsonl"

        # Open files in append mode
        self.text_file = open(self.text_path, "a")
        self.json_file = open(self.json_path, "a")

    # Timestamp helper
    @staticmethod
    def _ts():
        """
        Return a human-readable timestamp.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Text logging
    def log_text(self, message: str):
        """
        Write a single line to the human-readable text log.

        Format:
            YYYY-MM-DD HH:MM:SS     message
        """
        line = f"{self._ts()}\t{message}\n"

        # write to file
        self.text_file.write(line)
        self.text_file.flush()

        # print to console
        print(line, end="")

    # JSON logging
    def log_json(self, **kwargs):
        """
        Write a structured JSON entry to run.jsonl.

        Each call writes one JSON object per line, e.g.:
            {"time": "...", "event": "batch_start", "batch": 3}

        This format is ideal for downstream parsing.
        """
        entry = {"time": self._ts(), **kwargs}
        line = json.dumps(entry) + "\n"

        # write to file
        self.json_file.write(line)
        self.json_file.flush()

    # Cleanup
    def close(self):
        """
        Close the logger and flush all files and directories.
        """
        self.text_file.close()
        self.json_file.close()