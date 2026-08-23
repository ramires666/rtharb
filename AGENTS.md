# Agent Rules & Operational Standards

## 1. Process and Task Hygiene
- Always clean up background tasks before completing a turn.
- Ensure 0 background tasks in `manage_task: list` unless explicitly instructed to keep a service running.
- Prevent CPU/memory hogging by terminating completed processes.
- Use sub-agents aggressively when independent work can run in parallel or reduce total cost/latency. Cheap agents are preferred for bounded routine work, but any reasoning level is allowed when it improves speed or reliability. Avoid delegation only when tasks would contend on the same files or delegation overhead exceeds the expected gain.

## 2. Deliverable Verification & Pre-Flight Gatekeeper
- Always verify file content and size using `view_file` or direct inspection before informing the user of completion.
- Ensure HTML one-pagers and dashboards have fully rendered data, charts, and scripts.
- Run `audit_integrity.py` to assert data integrity before finishing.

## 3. Encoding & Compatibility
- Enforce UTF-8 console output in all Python scripts on Windows.
- Always provide simple `.bat` 1-click launchers for easy user execution.

## 4. Raw Historical Data Integrity
- Absolute prohibition on mock, synthetic, procedural, or keyframe-interpolated quotes in charts and backtests.
- Always export and render raw 1-minute historical data straight from Alpaca Parquet archives.
