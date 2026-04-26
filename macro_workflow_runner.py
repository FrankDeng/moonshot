import argparse
import json
import os
from datetime import datetime
from typing import Any

import export_analyst_result as ear
import macro_indicator as mi
import requests
from psycopg2.extras import Json, RealDictCursor
import html


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


def _build_data_summary_for_llm(data: Any, max_rows: int = 80) -> str:
    """
    Build a compact CSV-like summary to control token usage.
    """
    try:
        if data is None or not hasattr(data, "head"):
            return "No tabular data available."
        cols = list(getattr(data, "columns", []))
        tail_n = max(10, min(20, max_rows // 4))
        head_n = max(20, max_rows - tail_n)
        head = data.head(head_n)
        tail = data.tail(tail_n)
        text = []
        text.append(f"columns={cols}")
        text.append(f"rows={len(data)}")
        text.append("head:")
        text.append(head.to_csv(index=False))
        text.append("tail:")
        text.append(tail.to_csv(index=False))
        return "\n".join(text)
    except Exception as exc:
        return f"Failed to build data summary: {exc}"


def _analyze_with_deepseek(
    description: str,
    left: str,
    right: str,
    data: Any,
    ai_cfg: dict[str, Any],
) -> str:
    api_key_env = str(ai_cfg.get("api_key_env") or "DEEPSEEK_API_KEY")
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing DeepSeek API key env: {api_key_env}")

    model = str(ai_cfg.get("model") or "deepseek-chat")
    base_url = str(ai_cfg.get("base_url") or "https://api.deepseek.com").rstrip("/")
    temperature = float(ai_cfg.get("temperature") or 0.2)
    max_tokens = int(ai_cfg.get("max_tokens") or 1200)
    max_rows = int(ai_cfg.get("max_rows") or 200)

    data_summary = _build_data_summary_for_llm(data, max_rows=max_rows)

    system_prompt = (
        "You are a macro strategist. "
        "Given transformed cycle data, provide concise and actionable macro-cycle diagnosis "
        "and high-level cross-asset direction view."
    )
    user_prompt = f"""
Workflow description:
{description}

Pair:
left={left}
right={right}

Processed data summary:
{data_summary}

Please provide:
1) Current cycle phase and confidence
2) Main macro drivers inferred from the series dynamics
3) 1-3 month directional view for major asset classes (equities, rates, FX, commodities)
4) Key risks and invalidation signals

Keep answer structured with short headings and bullet points.
""".strip()

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(f"{base_url}/chat/completions", headers=headers, json=payload, timeout=120)
    if resp.status_code >= 300:
        raise RuntimeError(f"DeepSeek API error {resp.status_code}: {resp.text[:500]}")
    body = resp.json()
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("DeepSeek API returned empty choices.")
    content = choices[0].get("message", {}).get("content", "")
    if not content:
        raise RuntimeError("DeepSeek API returned empty content.")
    return str(content).strip()


def _ensure_indicators_table(conn: Any) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS indicators (
                id SERIAL PRIMARY KEY,
                indicator_name VARCHAR(255) NOT NULL,
                info TEXT,
                data JSONB,
                summary TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
    conn.commit()


def _iso_or_none(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.isoformat()
    try:
        return str(v)
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    """
    Recursively convert python/pandas/numpy objects to JSON-serializable types.
    """
    # primitives
    if value is None or isinstance(value, (str, int, bool, float)):
        # normalize NaN/Inf to None for strict JSON compatibility
        if isinstance(value, float):
            if value != value or value in (float("inf"), float("-inf")):
                return None
        return value

    # datetime-like
    if isinstance(value, datetime):
        return value.isoformat()
    try:
        # date type check without importing date explicitly
        if hasattr(value, "isoformat") and value.__class__.__name__ in {"date", "Timestamp", "datetime64", "datetime"}:
            return value.isoformat()
    except Exception:
        pass

    # pandas Timestamp / numpy scalars
    try:
        import pandas as pd  # type: ignore

        if isinstance(value, pd.Timestamp):
            return value.isoformat()
    except Exception:
        pass

    # mappings
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    # iterables
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    # fallback
    try:
        return str(value)
    except Exception:
        return None


def _write_pages_site(
    *,
    out_root: str,
    workflow_name: str,
    workflow_path: str,
    executed_jobs: list[dict[str, Any]],
) -> None:
    """
    Generate a minimal static site under out_root for GitHub Pages.
    - index.html lists jobs and links
    - each job page shows description, summary, and embeds the plotly HTML via iframe
    """
    os.makedirs(out_root, exist_ok=True)
    index_items = []
    for job in executed_jobs:
        job_id = str(job.get("id"))
        indicator_name = str(job.get("indicator_name") or job_id)
        left = str(job.get("left") or "")
        right = str(job.get("right") or "")
        desc = str(job.get("description") or "")
        skipped = bool(job.get("skipped", False))
        page_name = f"{job_id}.page.html"
        index_items.append(
            f"<li><a href='{html.escape(page_name)}'>{html.escape(indicator_name)}</a> "
            f"({html.escape(left)} vs {html.escape(right)}) "
            f"{'[SKIPPED]' if skipped else ''}<br/><small>{html.escape(desc)}</small></li>"
        )

        fig_path = str(job.get("figure") or "")
        analysis_path = str(job.get("analysis") or "")
        summary_text = ""
        if analysis_path and os.path.exists(analysis_path):
            with open(analysis_path, "r", encoding="utf-8") as fp:
                summary_text = fp.read()

        fig_file = os.path.basename(fig_path) if fig_path else ""
        iframe_html = (
            f"<iframe src='{html.escape(fig_file)}' style='width:100%;height:720px;border:0;'></iframe>"
            if fig_file
            else "<p><em>No figure generated.</em></p>"
        )

        page_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(indicator_name)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
    .meta {{ color: #475569; }}
    pre {{ background: #0b1020; color: #e5e7eb; padding: 14px; overflow-x: auto; border-radius: 10px; }}
    a {{ color: #2563eb; text-decoration: none; }}
  </style>
</head>
<body>
  <p><a href="index.html">← Back</a></p>
  <h1>{html.escape(indicator_name)}</h1>
  <p class="meta"><strong>Workflow</strong>: {html.escape(workflow_name)} | <code>{html.escape(workflow_path)}</code></p>
  <p><strong>Pair</strong>: {html.escape(left)} vs {html.escape(right)}</p>
  <p><strong>Description</strong>: {html.escape(desc)}</p>
  <h2>Chart</h2>
  {iframe_html}
  <h2>LLM Summary</h2>
  <pre>{html.escape(summary_text or '(empty)')}</pre>
</body>
</html>
"""
        with open(os.path.join(out_root, page_name), "w", encoding="utf-8") as fp:
            fp.write(page_html)

    index_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(workflow_name)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif; margin: 24px; }}
    li {{ margin-bottom: 14px; }}
    small {{ color: #475569; }}
  </style>
</head>
<body>
  <h1>{html.escape(workflow_name)}</h1>
  <p><code>{html.escape(workflow_path)}</code></p>
  <h2>Indicators</h2>
  <ol>
    {''.join(index_items)}
  </ol>
</body>
</html>
"""
    with open(os.path.join(out_root, "index.html"), "w", encoding="utf-8") as fp:
        fp.write(index_html)


def _get_source_max_date(conn: Any, source: str, ticker: str, requested_date_col: str | None) -> datetime | None:
    with conn.cursor() as cur:
        if source == "macro_data_revisions":
            if requested_date_col == "eco_release_dt":
                cur.execute(
                    """
                    SELECT MAX(COALESCE(eco_release_dt, data_date::timestamptz))
                    FROM macro_data_revisions
                    WHERE is_latest = TRUE AND ticker = %s
                    """,
                    (ticker,),
                )
            else:
                cur.execute(
                    """
                    SELECT MAX(data_date::timestamptz)
                    FROM macro_data_revisions
                    WHERE is_latest = TRUE AND ticker = %s
                    """,
                    (ticker,),
                )
            row = cur.fetchone()
            return row[0] if row else None

        if source == "market_prices":
            cur.execute(
                """
                SELECT MAX(price_date::timestamptz)
                FROM market_prices
                WHERE ticker = %s
                """,
                (ticker,),
            )
            row = cur.fetchone()
            return row[0] if row else None

    raise RuntimeError(f"Unsupported source for max-date check: {source}")


def _get_existing_indicator(conn: Any, indicator_name: str) -> dict[str, Any] | None:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            """
            SELECT id, indicator_name, info, data, summary, created_at
            FROM indicators
            WHERE indicator_name = %s
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (indicator_name,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def _upsert_indicator(
    conn: Any,
    *,
    indicator_name: str,
    info: str,
    data_payload: dict[str, Any],
    summary: str,
) -> int:
    existing = _get_existing_indicator(conn, indicator_name)
    with conn.cursor() as cur:
        if existing:
            cur.execute(
                """
                UPDATE indicators
                SET info = %s, data = %s, summary = %s, created_at = CURRENT_TIMESTAMP
                WHERE id = %s
                """,
                (info, Json(data_payload), summary, int(existing["id"])),
            )
            conn.commit()
            return int(existing["id"])
        cur.execute(
            """
            INSERT INTO indicators (indicator_name, info, data, summary)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (indicator_name, info, Json(data_payload), summary),
        )
        new_id = int(cur.fetchone()[0])
    conn.commit()
    return new_id


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
    ai_defaults = defaults.get("ai", {})
    publish_defaults = defaults.get("publish", {})
    if not isinstance(diff_defaults, dict) or not isinstance(cycle_defaults, dict) or not isinstance(output_defaults, dict):
        raise RuntimeError("`defaults.diff`, `defaults.cycle`, `defaults.output` must be objects.")
    if not isinstance(ai_defaults, dict):
        raise RuntimeError("`defaults.ai` must be an object.")
    if not isinstance(publish_defaults, dict):
        raise RuntimeError("`defaults.publish` must be an object.")

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
        _ensure_indicators_table(conn)
        workflow_name = str(cfg.get("name") or "macro-workflow")

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

            left_source = str(diff_params.get("left_source") or "macro_data_revisions")
            right_source = str(diff_params.get("right_source") or "macro_data_revisions")
            left_date_col = diff_params.get("left_date_col")
            right_date_col = diff_params.get("right_date_col")
            left_max = _get_source_max_date(conn, left_source, left, left_date_col if isinstance(left_date_col, str) else None)
            right_max = _get_source_max_date(conn, right_source, right, right_date_col if isinstance(right_date_col, str) else None)
            source_max_date = max([d for d in [left_max, right_max] if d is not None], default=None)

            indicator_name = str(job.get("indicator_name") or job_id)
            existing = _get_existing_indicator(conn, indicator_name)
            existing_max = None
            if existing and isinstance(existing.get("data"), dict):
                existing_max_raw = existing["data"].get("source_max_date")
                if existing_max_raw:
                    try:
                        existing_max = datetime.fromisoformat(str(existing_max_raw).replace("Z", "+00:00"))
                    except Exception:
                        existing_max = None

            if existing and source_max_date is not None and existing_max is not None and source_max_date <= existing_max:
                executed.append(
                    {
                        "id": job_id,
                        "indicator_name": indicator_name,
                        "left": left,
                        "right": right,
                        "skipped": True,
                        "reason": "no_new_data_points",
                        "source_max_date": _iso_or_none(source_max_date),
                        "existing_source_max_date": _iso_or_none(existing_max),
                    }
                )
                continue

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

            analysis_path = None
            description = str(job.get("description") or "")
            analysis = ""
            ai_enabled_default = bool(ai_defaults.get("enabled", False))
            ai_enabled_job = job.get("analyze_with_deepseek")
            ai_enabled = ai_enabled_default if ai_enabled_job is None else bool(ai_enabled_job)
            if ai_enabled:
                ai_cfg = _merge_dict(ai_defaults, job.get("ai", {}) if isinstance(job.get("ai"), dict) else {})
                analysis = _analyze_with_deepseek(
                    description=description or f"Cycle analysis for {left} vs {right}",
                    left=left,
                    right=right,
                    data=data,
                    ai_cfg=ai_cfg,
                )
                analysis_name = str(job.get("analysis") or f"{job_id}_analysis.md")
                analysis_path = os.path.join(out_root, analysis_name)
                os.makedirs(os.path.dirname(analysis_path), exist_ok=True)
                with open(analysis_path, "w", encoding="utf-8") as fp:
                    fp.write(analysis)

            publish_logs = []
            publish_enabled_default = bool(publish_defaults.get("enabled", False))
            publish_job_cfg = job.get("publish")
            if isinstance(publish_job_cfg, dict):
                publish_enabled = bool(publish_job_cfg.get("enabled", publish_enabled_default))
            else:
                # backward compatible: allow boolean flag
                publish_enabled = publish_enabled_default if publish_job_cfg is None else bool(publish_job_cfg)
            if publish_enabled:
                channels_raw = None
                if isinstance(publish_job_cfg, dict):
                    channels_raw = publish_job_cfg.get("channels")
                if channels_raw is None:
                    channels_raw = job.get("publish_channels", publish_defaults.get("channels", ["x"]))
                if isinstance(channels_raw, str):
                    channels = [c.strip() for c in channels_raw.split(",") if c.strip()]
                elif isinstance(channels_raw, list):
                    channels = [str(c).strip() for c in channels_raw if str(c).strip()]
                else:
                    channels = ["x"]
                dry_run_default = bool(publish_defaults.get("dry_run", False))
                dry_run_job = None
                if isinstance(publish_job_cfg, dict):
                    dry_run_job = publish_job_cfg.get("dry_run")
                if dry_run_job is None:
                    dry_run_job = job.get("publish_dry_run")
                dry_run = dry_run_default if dry_run_job is None else bool(dry_run_job)

                result_obj = ear.AnalystResult(
                    indicator_name=indicator_name,
                    workflow_name=workflow_name,
                    description=description,
                    summary=analysis or "",
                    generated_at=datetime.utcnow().isoformat(),
                    figure_path=fig_path,
                    analysis_path=analysis_path,
                )
                publish_result = ear.publish_single_result(channels=channels, result=result_obj, dry_run=dry_run)
                publish_logs = publish_result.get("published", []) if isinstance(publish_result, dict) else []

            data_payload = {
                "workflow_name": workflow_name,
                "workflow_path": workflow_path,
                "job_id": job_id,
                "description": description,
                "left": left,
                "right": right,
                "source_max_date": _iso_or_none(source_max_date),
                "generated_at": datetime.utcnow().isoformat(),
                "figure_path": fig_path,
                "csv_path": csv_path,
                "analysis_path": analysis_path,
                "row_count": int(len(data)),
                "columns": list(getattr(data, "columns", [])),
                "data_records": _json_safe(data.to_dict(orient="records")),
                "publish_logs": _json_safe(publish_logs),
            }
            indicator_id = _upsert_indicator(
                conn,
                indicator_name=indicator_name,
                info=description or f"{workflow_name}::{job_id}",
                data_payload=data_payload,
                summary=analysis,
            )

            executed.append(
                {
                    "id": job_id,
                    "indicator_id": indicator_id,
                    "indicator_name": indicator_name,
                    "left": left,
                    "right": right,
                    "description": description,
                    "figure": fig_path,
                    "csv": csv_path,
                    "analysis": analysis_path,
                    "published": bool(publish_logs),
                    "publish_logs": _json_safe(publish_logs),
                    "skipped": False,
                }
            )
    finally:
        conn.close()

    # Optional: GitHub Pages output
    pages_cfg = output_defaults.get("pages", {})
    if pages_cfg is None:
        pages_cfg = {}
    if not isinstance(pages_cfg, dict):
        raise RuntimeError("`defaults.output.pages` must be an object.")

    pages_enabled = bool(pages_cfg.get("enabled", False))
    if pages_enabled:
        _write_pages_site(
            out_root=out_root,
            workflow_name=workflow_name,
            workflow_path=workflow_path,
            executed_jobs=executed,
        )

    return {"ok": True, "workflow": workflow_path, "mode": "batch", "executed_jobs": executed, "pages_enabled": pages_enabled}


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
