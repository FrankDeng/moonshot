import argparse
import json
import os
from typing import Any

import macro_indicator as mi


def _resolve_refs(value: Any, ctx: dict[str, Any]) -> Any:
    """
    Resolve references in params:
    - "$name" -> ctx["name"]
    - recursive for list/dict
    """
    if isinstance(value, str) and value.startswith("$"):
        key = value[1:]
        if key not in ctx:
            raise RuntimeError(f"Reference not found in context: {value}")
        return ctx[key]
    if isinstance(value, list):
        return [_resolve_refs(v, ctx) for v in value]
    if isinstance(value, dict):
        return {k: _resolve_refs(v, ctx) for k, v in value.items()}
    return value


def _normalize_result(res: Any) -> Any:
    """
    Normalize function results so they can be referenced consistently in workflow.
    """
    # cycle_analysis currently returns (data, fig)
    if isinstance(res, tuple) and len(res) == 2:
        return {"data": res[0], "figure": res[1]}
    return res


def _save_artifacts(step: dict[str, Any], result: Any, output_dir: str) -> None:
    if not isinstance(result, dict):
        return

    if "save_figure_html" in step:
        fig_key = step.get("figure_key", "figure")
        fig = result.get(fig_key)
        if fig is None:
            raise RuntimeError(f"Step asks save_figure_html but key not found: {fig_key}")
        out_path = os.path.join(output_dir, step["save_figure_html"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.write_html(out_path)

    if "save_data_csv" in step:
        data_key = step.get("data_key", "data")
        data = result.get(data_key)
        if data is None:
            raise RuntimeError(f"Step asks save_data_csv but key not found: {data_key}")
        out_path = os.path.join(output_dir, step["save_data_csv"])
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        data.to_csv(out_path, index=False)


def _build_op_registry() -> dict[str, Any]:
    return {
        "get_neon_conn": mi.get_neon_conn,
        "fetch_market_prices": mi.fetch_market_prices,
        "fetch_macro_latest": mi.fetch_macro_latest,
        "diff_by_tickers": mi.diff_by_tickers,
        "compute_macro_surprise": mi.compute_macro_surprise,
        "add_zscore": mi.add_zscore,
        "pivot_prices": mi.pivot_prices,
        "plot_timeseries": mi.plot_timeseries,
        "plot_macro_surprise": mi.plot_macro_surprise,
        "cycle_analysis": mi.cycle_analysis,
    }


def run_workflow(workflow_path: str, output_dir: str = ".") -> dict[str, Any]:
    with open(workflow_path, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    if not isinstance(cfg, dict):
        raise RuntimeError("Workflow JSON must be an object.")

    # New concise/batch format
    if isinstance(cfg.get("jobs"), list):
        return run_workflow_batch(cfg, workflow_path=workflow_path, output_dir=output_dir)

    # Backward-compatible step-by-step format
    steps = cfg.get("steps")
    if not isinstance(steps, list) or not steps:
        raise RuntimeError("Workflow JSON requires `steps` (legacy) or `jobs` (batch).")

    op_registry = _build_op_registry()

    ctx: dict[str, Any] = {}
    executed: list[dict[str, Any]] = []

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            raise RuntimeError(f"Step[{idx}] must be an object.")
        op = step.get("op")
        if not isinstance(op, str) or op not in op_registry:
            raise RuntimeError(f"Step[{idx}] invalid op: {op}")

        step_id = str(step.get("id") or f"step_{idx}")
        raw_params = step.get("params", {})
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            raise RuntimeError(f"Step[{idx}] params must be object.")

        params = _resolve_refs(raw_params, ctx)
        result = _normalize_result(op_registry[op](**params))
        ctx[step_id] = result
        executed.append({"id": step_id, "op": op})

        _save_artifacts(step, result, output_dir=output_dir)

    # close connection if present
    for v in ctx.values():
        if hasattr(v, "close") and callable(getattr(v, "close")):
            try:
                v.close()
            except Exception:
                pass

    return {"ok": True, "workflow": workflow_path, "executed_steps": executed}


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(override)
    return out


def run_workflow_batch(cfg: dict[str, Any], workflow_path: str, output_dir: str = ".") -> dict[str, Any]:
    """
    Concise batch format:
    {
      "connection": {"neon_config_path": "..."},
      "defaults": {
        "diff": {...},
        "cycle": {...},
        "output": {"dir": "outputs"}
      },
      "jobs": [
        {"id": "m2_gdp", "left": "...", "right": "..."},
        {"id": "cpi_pmi", "left": "...", "right": "...", "cycle": {"period_months": 18}}
      ]
    }
    """
    jobs = cfg.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError("Batch format requires non-empty `jobs` array.")

    defaults = cfg.get("defaults", {})
    if defaults is None:
        defaults = {}
    if not isinstance(defaults, dict):
        raise RuntimeError("`defaults` must be an object.")

    diff_defaults = defaults.get("diff", {})
    cycle_defaults = defaults.get("cycle", {})
    output_defaults = defaults.get("output", {})
    if not isinstance(diff_defaults, dict) or not isinstance(cycle_defaults, dict) or not isinstance(output_defaults, dict):
        raise RuntimeError("`defaults.diff`, `defaults.cycle`, `defaults.output` must be objects.")

    conn_cfg = cfg.get("connection", {})
    if conn_cfg is None:
        conn_cfg = {}
    if not isinstance(conn_cfg, dict):
        raise RuntimeError("`connection` must be an object.")

    out_subdir = str(output_defaults.get("dir") or "outputs").strip()
    out_root = os.path.join(output_dir, out_subdir)
    os.makedirs(out_root, exist_ok=True)

    conn = mi.get_neon_conn(**conn_cfg)
    executed: list[dict[str, Any]] = []
    try:
        for i, job in enumerate(jobs):
            if not isinstance(job, dict):
                raise RuntimeError(f"jobs[{i}] must be an object.")
            job_id = str(job.get("id") or f"job_{i}")
            left = job.get("left")
            right = job.get("right")
            if not isinstance(left, str) or not isinstance(right, str):
                raise RuntimeError(f"jobs[{i}] requires string `left` and `right`.")

            diff_params = _merge_dict(
                diff_defaults,
                {
                    "left": left,
                    "right": right,
                },
            )
            diff_params = _merge_dict(diff_params, job.get("diff", {}) if isinstance(job.get("diff"), dict) else {})
            diff_params["conn"] = conn

            spread_df = mi.diff_by_tickers(**diff_params)

            cycle_params = _merge_dict(
                cycle_defaults,
                {
                    "df": spread_df,
                    "date_col": "date",
                    "value_col": f"{left}-{right}",
                },
            )
            cycle_params = _merge_dict(cycle_params, job.get("cycle", {}) if isinstance(job.get("cycle"), dict) else {})

            cycle_res = _normalize_result(mi.cycle_analysis(**cycle_params))
            fig = cycle_res["figure"]
            data = cycle_res["data"]

            fig_name = str(job.get("figure") or f"{job_id}.html")
            csv_name = str(job.get("csv") or f"{job_id}.csv")
            fig_path = os.path.join(out_root, fig_name)
            csv_path = os.path.join(out_root, csv_name)
            os.makedirs(os.path.dirname(fig_path), exist_ok=True)
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            fig.write_html(fig_path)
            data.to_csv(csv_path, index=False)

            executed.append({"id": job_id, "left": left, "right": right, "figure": fig_path, "csv": csv_path})
    finally:
        conn.close()

    return {"ok": True, "workflow": workflow_path, "mode": "batch", "executed_jobs": executed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run macro workflow from JSON.")
    parser.add_argument(
        "--workflow",
        default="macro_workflow_example.json",
        help="Path to workflow JSON file. Default: macro_workflow_example.json",
    )
    parser.add_argument("--output-dir", default=".", help="Directory for output artifacts.")
    args = parser.parse_args()

    if not os.path.exists(args.workflow):
        raise RuntimeError(
            f"Workflow file not found: {args.workflow}. "
            "Use --workflow <path> or place macro_workflow_example.json in workspace root."
        )

    res = run_workflow(workflow_path=args.workflow, output_dir=args.output_dir)
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
