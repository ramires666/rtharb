# Mandatory Agent Rules & Execution Guidelines

---

## 1. Strict Process & Background Task Hygiene (Zero Lingering Tasks)
- **Automatic Task Cleanup:** NEVER leave finished, hanging, or orphaned background tasks running in the system.
- **Process Verification:** After running background computations (such as backtests, optimizers, or servers), immediately terminate completed tasks with `manage_task(Action='kill')` or ensure process completion.
- **Pre-Turn Checklist:** Before concluding any response, ALWAYS verify that `manage_task(Action='list')` has **0 active background tasks** (unless the user explicitly asked for a continuous background daemon).
- **No CPU/RAM Leaks:** Ensure no orphaned Python, Streamlit, or node processes remain running in the OS after computations finish.

---

## 2. Mandatory Verification of Deliverables Before Reporting
- **Never Assume File Contents:** NEVER report to the user that a report, script, or application is ready without explicitly verifying the file content and non-zero file size on disk.
- **Verify HTML/Visual One-Pagers:** For HTML reports or interactive one-pagers, inspect lines from the file to guarantee that full charts, scripts, and embedded data payloads are present (not placeholder stubs).
- **End-to-End Sanity Check:** Ensure all batch files (`.bat`), scripts, and links actually point to existing, functional files.

---

## 3. Reliable File Persistence & Tool Usage
- **Direct Workspace Writes:** Always write deliverable artifacts directly to the workspace using verified write tools (`write_to_file`) to prevent data loss or transaction rollback.
- **Encoding Safety:** Reconfigure standard output to UTF-8 (`sys.stdout.reconfigure(encoding='utf-8')`) in all Python CLI entry points on Windows to prevent `UnicodeEncodeError` with special characters and emojis.

---

## 4. Multi-Year Historical Data Best Practices
- **1-Minute Data Depth:** Yahoo Finance is limited to 30 days of 1-minute history. For multi-year intraday research (1–7+ years), always utilize Alpaca Market Data API (`alpaca-py`), local Parquet cache, or custom CSV/Parquet archives.

---

## 5. Absolute Prohibition on Synthetic or Procedural Quotes
- **Zero Mock / Synthetic Data:** NEVER generate, simulate, approximate, or interpolate candlestick prices or time series using procedural functions, random walks (`rand()`, `Math.random()`), sine waves (`Math.sin()`), or piecewise keyframe approximations.
- **Raw Parquet/CSV Integrity:** ALL interactive charts, reports, backtests, and session inspectors MUST receive and display 100% genuine, raw historical bar arrays extracted directly from local Parquet/CSV archives (such as Alpaca 1-minute historical datasets).
- **Pre-Flight Audit Gatekeeper:** Always run `audit_integrity.py` before concluding any turn involving report deliverables to verify 100% compliance with real data and zero active tasks.
