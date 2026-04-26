import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Sequence
from sktime.transformations.series.detrend import Detrender
from sktime.forecasting.trend import PolynomialTrendForecaster
import numpy as np
import psycopg2
import matplotlib.pyplot as plt  # type: ignore


def _load_database_url(neon_config_path: str = "neon_connection_string.json") -> str:
    env_url = os.environ.get("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    if os.path.exists(neon_config_path):
        with open(neon_config_path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        if isinstance(loaded, dict):
            file_url = str(loaded.get("connection_string") or "").strip()
            if file_url:
                return file_url
    raise RuntimeError(f"Missing DATABASE_URL and `{neon_config_path}` connection_string.")


def get_neon_conn(neon_config_path: str = "neon_connection_string.json"):
    """
    Return a psycopg2 connection to Neon. Caller is responsible for closing it.
    """
    return psycopg2.connect(_load_database_url(neon_config_path))


def _as_list(values: str | Sequence[str]) -> list[str]:
    if isinstance(values, str):
        return [v.strip() for v in values.split(",") if v.strip()]
    return [str(v).strip() for v in values if str(v).strip()]


def fetch_market_prices(
    conn: Any,
    tickers: str | Sequence[str],
    start: date | str | None = None,
    end: date | str | None = None,
    columns: Sequence[str] | None = None,
):
    """
    Fetch `market_prices` into a pandas DataFrame.

    columns: subset of table columns to fetch (default: all numeric price columns + keys).
    """
    import pandas as pd  # type: ignore

    tickers_list = _as_list(tickers)
    if not tickers_list:
        raise RuntimeError("tickers is empty")

    def _d(x: date | str | None) -> date | None:
        if x is None:
            return None
        if isinstance(x, date) and not isinstance(x, datetime):
            return x
        if isinstance(x, datetime):
            return x.date()
        return datetime.fromisoformat(str(x)).date()

    start_d = _d(start)
    end_d = _d(end)

    with conn.cursor() as cur:
        if columns:
            safe_cols = ['"ticker"', '"price_date"'] + [f'"{c}"' for c in columns] + ['"updated_at"']
            select_cols = ", ".join(safe_cols)
        else:
            # default: fetch all columns
            select_cols = "*"

        where = ["ticker = ANY(%s)"]
        params: list[Any] = [tickers_list]
        if start_d is not None:
            where.append("price_date >= %s")
            params.append(start_d)
        if end_d is not None:
            where.append("price_date <= %s")
            params.append(end_d)

        cur.execute(
            f"""
            SELECT {select_cols}
            FROM market_prices
            WHERE {' AND '.join(where)}
            ORDER BY ticker, price_date
            """,
            tuple(params),
        )
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=colnames)
    if not df.empty:
        df["price_date"] = pd.to_datetime(df["price_date"])
    return df


def fetch_macro_latest(
    conn: Any,
    tickers: str | Sequence[str],
    start: date | str | None = None,
    end: date | str | None = None,
    columns: Sequence[str] | None = None,
):
    """
    Fetch latest macro values (is_latest=TRUE) from `macro_data_revisions`.
    Returns pandas DataFrame with `data_date` as datetime.
    """
    import pandas as pd  # type: ignore

    tickers_list = _as_list(tickers)
    if not tickers_list:
        raise RuntimeError("tickers is empty")

    def _d(x: date | str | None) -> date | None:
        if x is None:
            return None
        if isinstance(x, date) and not isinstance(x, datetime):
            return x
        if isinstance(x, datetime):
            return x.date()
        return datetime.fromisoformat(str(x)).date()

    start_d = _d(start)
    end_d = _d(end)

    with conn.cursor() as cur:
        if columns:
            safe_cols = ['"ticker"', '"data_date"'] + [f'"{c}"' for c in columns] + ['"eco_release_dt"', '"revision_num"', '"data_status"', '"created_at"']
            select_cols = ", ".join(dict.fromkeys(safe_cols))
        else:
            select_cols = "*"

        where = ["is_latest = TRUE", "ticker = ANY(%s)"]
        params: list[Any] = [tickers_list]
        if start_d is not None:
            where.append("data_date >= %s")
            params.append(start_d)
        if end_d is not None:
            where.append("data_date <= %s")
            params.append(end_d)

        cur.execute(
            f"""
            SELECT {select_cols}
            FROM macro_data_revisions
            WHERE {' AND '.join(where)}
            ORDER BY ticker, data_date
            """,
            tuple(params),
        )
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]

    df = pd.DataFrame(rows, columns=colnames)
    if not df.empty:
        df["data_date"] = pd.to_datetime(df["data_date"])
        if "eco_release_dt" in df.columns:
            df["eco_release_dt"] = pd.to_datetime(df["eco_release_dt"], errors="coerce", utc=True)
    return df


def align_union_ffill_and_diff(
    left,
    right,
    *,
    left_date_col: str = "date",
    left_value_col: str = "value",
    right_date_col: str = "date",
    right_value_col: str = "value",
    out_date_col: str = "date",
    out_left_col: str = "left",
    out_right_col: str = "right",
    out_diff_col: str = "diff",
):
    """
    计算两个时间序列的差值：diff = left - right。

    当两边日期/频率不一致时：
    - 取日期并集（不丢任何一边出现过的日期点）
    - 各自向后填充（ffill，只用过去值，不会“未来函数”）
    - 不删除/省略任何对齐后的日期点（开头没有历史的部分会保留 NaN）

    参数输入支持：
    - pandas.DataFrame（包含 date_col 与 value_col）
    - pandas.Series（DatetimeIndex）

    返回：
    - pandas.DataFrame：包含 [out_date_col, out_left_col, out_right_col, out_diff_col]
    """
    import pandas as pd  # type: ignore

    def _to_series(obj, date_col: str, value_col: str):
        if isinstance(obj, pd.Series):
            s = obj.copy()
            if not isinstance(s.index, pd.DatetimeIndex):
                s.index = pd.to_datetime(s.index)
            s = s.sort_index()
            return pd.to_numeric(s, errors="coerce")

        if not isinstance(obj, pd.DataFrame):
            raise RuntimeError("Input must be a pandas DataFrame or Series.")

        if date_col not in obj.columns or value_col not in obj.columns:
            raise RuntimeError(f"Missing columns: {date_col}, {value_col}")

        tmp = obj[[date_col, value_col]].copy()
        tmp[date_col] = pd.to_datetime(tmp[date_col])
        tmp = tmp.sort_values(date_col)
        # 同一天多条时，保留最后一条（不额外删点，只解决重复键）
        tmp = tmp.drop_duplicates(subset=[date_col], keep="last")
        s = pd.Series(pd.to_numeric(tmp[value_col], errors="coerce").values, index=tmp[date_col].values)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()

    left_s = _to_series(left, left_date_col, left_value_col)
    right_s = _to_series(right, right_date_col, right_value_col)

    union_index = left_s.index.union(right_s.index).sort_values()
    left_aligned = left_s.reindex(union_index).ffill()
    right_aligned = right_s.reindex(union_index).ffill()

    out = pd.DataFrame(
        {
            out_date_col: union_index,
            out_left_col: left_aligned.values,
            out_right_col: right_aligned.values,
            out_diff_col: (left_aligned - right_aligned).values,
        }
    )
    return out


def diff_by_tickers(
    conn: Any,
    *,
    left: str,
    right: str,
    left_source: str,
    right_source: str,
    left_date_col: str | None = None,
    right_date_col: str | None = None,
    left_value_col: str | None = None,
    right_value_col: str | None = None,
    start: date | str | None = None,
    end: date | str | None = None,
):
    """
    输入 left/right 为 ticker 名称，来源表由 left_source/right_source 指定：
    - 'market_prices'
    - 'macro_data_revisions'（仅取 is_latest=TRUE）

    自动取数后做日期并集 + ffill，对齐后计算 diff = left - right。
    """
    import pandas as pd  # type: ignore

    def _fetch_one(ticker: str, source: str, date_col: str | None, value_col: str | None):
        if source == "market_prices":
            vcol = value_col or "px_last"
            requested_dcol = date_col or "price_date"
            fallback_dcol = "price_date"
            df = fetch_market_prices(conn, [ticker], start=start, end=end, columns=[vcol])
            if df.empty:
                return pd.DataFrame({"date": [], "value": []})
            dcol = requested_dcol if requested_dcol in df.columns else fallback_dcol
            if dcol not in df.columns:
                raise RuntimeError(f"market_prices missing fallback date column: {fallback_dcol}")
            if vcol not in df.columns:
                raise RuntimeError(f"market_prices missing value column: {vcol}")
            df = df.rename(columns={dcol: "date", vcol: "value"})
            return df[["date", "value"]]

        if source == "macro_data_revisions":
            vcol = value_col or "px_last"
            requested_dcol = date_col or "data_date"
            # Always include data_date for fallback when using eco_release_dt
            cols = [vcol] + ([requested_dcol] if requested_dcol not in {"data_date"} else [])
            if requested_dcol == "eco_release_dt" and "data_date" not in cols:
                cols.append("data_date")
            df = fetch_macro_latest(conn, [ticker], start=start, end=end, columns=cols)
            if df.empty:
                return pd.DataFrame({"date": [], "value": []})
            if vcol not in df.columns:
                raise RuntimeError(f"macro_data_revisions missing value column: {vcol}")
            if requested_dcol == "eco_release_dt":
                # If eco_release_dt is missing/NULL, fall back to data_date (no dropping).
                if "data_date" not in df.columns:
                    raise RuntimeError("macro_data_revisions missing fallback column: data_date")
                if "eco_release_dt" in df.columns:
                    eco = pd.to_datetime(df["eco_release_dt"], errors="coerce")
                else:
                    eco = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
                fallback_date = pd.to_datetime(df["data_date"], errors="coerce")
                merged_date = fallback_date.copy()
                eco_mask = eco.notna()
                if eco_mask.any():
                    merged_date.loc[eco_mask] = eco.loc[eco_mask].values
                df["date"] = merged_date
                df["value"] = df[vcol]
            else:
                dcol = requested_dcol if requested_dcol in df.columns else "data_date"
                if dcol not in df.columns:
                    raise RuntimeError(f"macro_data_revisions missing fallback date column: data_date")
                df = df.rename(columns={dcol: "date", vcol: "value"})
            return df[["date", "value"]]

        raise RuntimeError("source must be 'market_prices' or 'macro_data_revisions'")

    left_df = _fetch_one(left, left_source, left_date_col, left_value_col)
    right_df = _fetch_one(right, right_source, right_date_col, right_value_col)

    out = align_union_ffill_and_diff(
        left_df,
        right_df,
        left_date_col="date",
        left_value_col="value",
        right_date_col="date",
        right_value_col="value",
        out_date_col="date",
        out_left_col=left,
        out_right_col=right,
        out_diff_col=f"{left}-{right}",
    )
    return out


def _causal_upper_winsor(s, q: float = 0.95):
    """
    One-sided (upper-tail) causal winsorization.
    At each t, cap by historical expanding q-quantile up to t.
    """
    cap = s.expanding(min_periods=12).quantile(q)
    cap = cap.ffill().bfill()
    return np.minimum(s, cap)


def _causal_dual_median_trend(s, period_months: int):
    """
    Robust single-sided trend:
    1) rolling median with long window
    2) second rolling median to increase stickiness
    """
    w1 = max(6, int(period_months * 3))
    w2 = max(6, int(period_months * 2))
    trend1 = s.rolling(window=w1, min_periods=max(6, period_months), center=False).median()
    trend1 = trend1.ffill().bfill()
    trend2 = trend1.rolling(window=w2, min_periods=max(6, period_months), center=False).median()
    return trend2.ffill().bfill()


def _apply_trend_momentum_cap(raw_log, trend, cap_ratio: float = 0.3):
    """
    Constrain |Δtrend| <= cap_ratio * |Δraw|
    to avoid trend inertia producing fake cycle jumps.
    """
    out = trend.copy()
    if len(out) < 2:
        return out

    raw_diff = raw_log.diff().fillna(0.0)
    tr_diff = out.diff().fillna(0.0)
    max_step = cap_ratio * raw_diff.abs()
    clamped_step = tr_diff.clip(lower=-max_step, upper=max_step)
    out.iloc[0] = out.iloc[0]
    for i in range(1, len(out)):
        out.iloc[i] = out.iloc[i - 1] + float(clamped_step.iloc[i])
    return out


def _classify_phase(z: float, dz: float, band: float = 0.05, high_band: float = 0.1) -> str:
    if np.isnan(z) or np.isnan(dz):
        return "Neutral"
    if z > band and dz > band:
        return "Expansion"
    if z > high_band and dz < -band:
        return "Slowdown"
    if z < -band and dz < -band:
        return "Contraction"
    if z < -band and dz > band:
        return "Recovery"
    return "Neutral"


def cycle_analysis(
    df,
    *,
    date_col: str,
    value_col: str,
    period_months: int = 24,
    winsor_q: float = 0.95,
    smooth_span: int = 3,
    hysteresis_band: float = 0.05,
):
    """
    Industrial-style cycle extraction (single-sided / causal):
    - one-sided outlier suppression
    - default path: log_winsor -> light smoothing + momentum cap
    - optional path: robust trend detrend
    - expanding std normalization with floor
    - light smoothing
    - phase classification + chart with phase backgrounds

    Returns:
      {
        "data": pd.DataFrame(...),
        "figure": plotly Figure
      }
    """
    import pandas as pd  # type: ignore
    import plotly.graph_objects as go  # type: ignore

    if date_col not in df.columns or value_col not in df.columns:
        raise RuntimeError(f"Missing columns: {date_col}, {value_col}")

    work = df[[date_col, value_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.sort_values(date_col).drop_duplicates(subset=[date_col], keep="last")
    work = work.reset_index(drop=True)

    # Preserve full date points; if value missing, carry forward causally.
    work["raw_value"] = work[value_col].astype(float)
    work["raw_value_ffill"] = work["raw_value"].ffill()

    # Log transform: macro-safe and variance stabilizing.
    # Use a signed log1p transform to preserve negative values:
    work["log_value"] = np.sign(work["raw_value_ffill"]) * np.log1p(np.abs(work["raw_value_ffill"]))

    # One-sided upper winsorization (causal).
    work["log_winsor"] = _causal_upper_winsor(work["log_value"], q=winsor_q)
    
    trend = _causal_dual_median_trend(work["log_winsor"], period_months=max(2, smooth_span))
    trend = _apply_trend_momentum_cap(work["log_winsor"], trend, cap_ratio=0.3)
    work["trend_log"] = trend

    # 1. 定义一个趋势拟合器 (比如 1 阶线性趋势，也可以用 2 阶多项式)
    forecaster = PolynomialTrendForecaster(degree=1)
    # 2. 包装进 Detrender
    transformer = Detrender(forecaster=forecaster)
    # 3. 拟合并提取趋势线
    # 这会返回去趋势后的残差 (即你的初步 Cycle)
    cycle_base = transformer.fit_transform(trend)
    # 如果想看提取出来的趋势线本身：
    work["cycle_base"] = cycle_base
    # Stable normalization: expanding std + floor.
    exp_std = work["cycle_base"].expanding(min_periods=max(6, period_months // 2)).std()
    global_std = float(work["cycle_base"].std(skipna=True)) if len(work) > 0 else 0.0
    floor = max(global_std * 0.5, 1e-6)
    denom = np.maximum(exp_std.fillna(floor), floor)
    work["cycle_z"] = work["cycle_base"] / denom

    # Very light smoothing (causal).
    work["cycle_final"] = work["cycle_z"].ewm(span=max(2, smooth_span), adjust=False).mean()
    work["cycle_d1"] = work["cycle_final"].diff()

    # Phase classification.
    work["phase"] = [
        _classify_phase(z, dz, band=hysteresis_band, high_band=hysteresis_band * 2)
        for z, dz in zip(work["cycle_final"], work["cycle_d1"])
    ]

    # Build chart with phase background.
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=work[date_col],
            y=work["raw_value"],
            mode="lines",
            name="raw_value",
            line=dict(color="#4b5563", width=2.2),
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=work[date_col],
            y=work["cycle_final"],
            mode="lines",
            name="cycle_final",
            line=dict(color="#2563eb", width=2.8),
            yaxis="y2",
        )
    )

    phase_colors = {
        "Expansion": "rgba(22,163,74,0.24)",
        "Slowdown": "rgba(245,158,11,0.24)",
        "Contraction": "rgba(220,38,38,0.22)",
        "Recovery": "rgba(37,99,235,0.22)",
        "Neutral": "rgba(100,116,139,0.18)",
    }

    # Add explicit legend items for phase colors (vrect has no native legend).
    phase_order = ["Expansion", "Slowdown", "Contraction", "Recovery", "Neutral"]
    for phase_name in phase_order:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=12, color=phase_colors[phase_name], symbol="square"),
                name=f"Phase: {phase_name}",
                legendgroup=f"phase_{phase_name}",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    def _phase_rect_bounds(ts):
        # Use midpoint boundaries to ensure continuous shading between sparse observations.
        if len(ts) == 1:
            x0 = ts.iloc[0] - pd.Timedelta(days=15)
            x1 = ts.iloc[0] + pd.Timedelta(days=15)
            return [x0], [x1]
        left_bounds = []
        right_bounds = []
        for i in range(len(ts)):
            if i == 0:
                left = ts.iloc[i] - (ts.iloc[i + 1] - ts.iloc[i]) / 2
            else:
                left = ts.iloc[i - 1] + (ts.iloc[i] - ts.iloc[i - 1]) / 2
            if i == len(ts) - 1:
                right = ts.iloc[i] + (ts.iloc[i] - ts.iloc[i - 1]) / 2
            else:
                right = ts.iloc[i] + (ts.iloc[i + 1] - ts.iloc[i]) / 2
            left_bounds.append(left)
            right_bounds.append(right)
        return left_bounds, right_bounds

    # Segment backgrounds by contiguous phase blocks, with midpoint bounds to avoid white gaps.
    if not work.empty:
        x_series = work[date_col].reset_index(drop=True)
        left_bounds, right_bounds = _phase_rect_bounds(x_series)
        start_idx = 0
        cur_phase = work.loc[0, "phase"]
        for i in range(1, len(work)):
            p = work.loc[i, "phase"]
            if p != cur_phase:
                fig.add_vrect(
                    x0=left_bounds[start_idx],
                    x1=right_bounds[i - 1],
                    fillcolor=phase_colors.get(cur_phase, phase_colors["Neutral"]),
                    opacity=1.0,
                    layer="below",
                    line_width=0,
                )
                start_idx = i
                cur_phase = p
        fig.add_vrect(
            x0=left_bounds[start_idx],
            x1=right_bounds[len(work) - 1],
            fillcolor=phase_colors.get(cur_phase, phase_colors["Neutral"]),
            opacity=1.0,
            layer="below",
            line_width=0,
        )

    fig.update_layout(
        title=f"Cycle Analysis: {value_col} (period={period_months}m)",
        xaxis_title="Date",
        yaxis=dict(title="Raw Value", side="left", title_font=dict(size=16), tickfont=dict(size=13)),
        yaxis2=dict(
            title="Cycle (normalized)",
            overlaying="y",
            side="right",
            showgrid=False,
            title_font=dict(size=16),
            tickfont=dict(size=13),
        ),
        xaxis=dict(title_font=dict(size=16), tickfont=dict(size=13)),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="left",
            x=0,
            font=dict(size=13),
            itemsizing="constant",
        ),
        title_font=dict(size=22),
        font=dict(size=14),
        autosize=False,
        width=1200,
        height=750,
        margin=dict(l=70, r=70, t=70, b=110),
        plot_bgcolor="white",
        paper_bgcolor="white",
        template="plotly_white",
    )

    return work,fig
if __name__ == "__main__":
    out = diff_by_tickers(
        conn=get_neon_conn(),
        left="M2% YOY Index",
        right="GDP CYOY Index",
        left_date_col="eco_release_dt",
        right_date_col="eco_release_dt",
        left_source="macro_data_revisions",
        right_source="macro_data_revisions",
        left_value_col="px_last",
        right_value_col="px_last",
    )
    df_out, fig = cycle_analysis(out, date_col="date", value_col="M2% YOY Index-GDP CYOY Index")
