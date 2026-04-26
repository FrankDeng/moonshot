import argparse
import json
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import psycopg2
from psycopg2.extras import execute_values


@dataclass
class BloombergRecord:
    ticker: str
    field: str
    as_of_date: date
    value: float | None
    currency: str | None
    source: str
    raw_payload: dict[str, Any]


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _load_database_url(neon_config_path: str = "neon_connection_string.json") -> str:
    # Keep the same priority as typical deployments: env first, file second.
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

    raise RuntimeError(
        "Missing DATABASE_URL and no valid `connection_string` found in "
        f"{neon_config_path}"
    )


def _parse_iso_date(value: str) -> date:
    raw = value.strip()
    if len(raw) == 8 and raw.isdigit():
        return datetime.strptime(raw, "%Y%m%d").date()
    return datetime.fromisoformat(raw).date()


def load_download_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, dict):
        raise RuntimeError("Config JSON must be an object.")
    return data


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_optional_timestamptz(value: Any) -> datetime | None:
    """
    pdblp can return ECO_RELEASE_DT as str/int/float/Timestamp/datetime.
    Normalize to python datetime (timezone-aware when possible) and avoid pandas nanosecond warnings.
    """
    if value is None:
        return None
    # NaN
    if isinstance(value, float) and value != value:
        return None

    # Common pdblp shape: YYYYMMDD.0 as float
    if isinstance(value, float):
        if value == 0.0:
            return None
        try:
            as_int = int(value)
            # guard against non-integer floats
            if abs(value - float(as_int)) < 1e-6:
                s = str(as_int)
                if len(s) >= 8 and s[:8].isdigit():
                    d = datetime.strptime(s[:8], "%Y%m%d").date()
                    return datetime.combine(d, datetime.min.time())
        except Exception:
            pass

    # Already datetime
    if isinstance(value, datetime):
        return value

    # date only
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())

    # Common: yyyymmdd as int/str
    if isinstance(value, int):
        if value == 0:
            return None
        s = str(value)
        if len(s) >= 8 and s[:8].isdigit():
            try:
                d = datetime.strptime(s[:8], "%Y%m%d").date()
                return datetime.combine(d, datetime.min.time())
            except Exception:
                pass

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s in {"0", "00000000"}:
            return None
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) >= 8:
            try:
                d = datetime.strptime(digits[:8], "%Y%m%d").date()
                return datetime.combine(d, datetime.min.time())
            except Exception:
                pass
        try:
            return datetime.fromisoformat(s)
        except Exception:
            return None

    # Fallback: pandas Timestamp / numpy datetime64, etc.
    try:
        import pandas as pd  # type: ignore

        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if ts is pd.NaT:
            return None
        # Remove nanoseconds to avoid: "Discarding nonzero nanoseconds in conversion."
        try:
            ts = ts.floor("s")
        except Exception:
            pass
        try:
            return ts.to_pydatetime()
        except TypeError:
            # pandas Timestamp.to_pydatetime(warn=False) on some versions
            return ts.to_pydatetime(warn=False)  # type: ignore[call-arg]
    except Exception:
        return None


def _to_snake_identifier(name: str) -> str:
    raw = name.strip().lower()
    out: list[str] = []
    prev_underscore = False
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                out.append("_")
                prev_underscore = True
    ident = "".join(out).strip("_")
    if not ident:
        raise RuntimeError(f"Invalid identifier from: {name!r}")
    if ident[0].isdigit():
        ident = f"f_{ident}"
    return ident


def _price_field_to_column(field: str) -> str:
    f = field.strip().upper()
    if f.startswith("PX_"):
        return _to_snake_identifier(f)
    return f"bbg_{_to_snake_identifier(f)}"


def _macro_field_to_column(field: str) -> str:
    f = field.strip().upper()
    if f == "ECO_RELEASE_DT":
        return "eco_release_dt"
    if f == "BN_SURVEY_MEDIAN":
        return "bn_survey_median"
    if f == "PX_LAST":
        return "px_last"
    return _to_snake_identifier(f)


def _ensure_columns_exist(conn: Any, table: str, columns: dict[str, str]) -> None:
    """
    columns: {column_name: sql_type}
    """
    with conn.cursor() as cur:
        for col, col_type in columns.items():
            if not col.replace("_", "").isalnum():
                raise RuntimeError(f"Unsafe column name: {col}")
            cur.execute(
                f'ALTER TABLE {table} ADD COLUMN IF NOT EXISTS "{col}" {col_type}'
            )
    conn.commit()


def _records_from_bdh_frame(
    frame: Any, requested_fields: list[str], default_ticker: str | None = None
) -> list[BloombergRecord]:
    """
    Convert common `blp` bdh output DataFrame into normalized records.
    Expected columns:
      - MultiIndex: (ticker, field)
      - or flat columns like "AAPL US Equity|PX_LAST"
    """
    records: list[BloombergRecord] = []
    try:
        columns = frame.columns
    except Exception as exc:
        raise RuntimeError("Unexpected bdh output: missing DataFrame-like columns") from exc

    is_multi = bool(getattr(columns, "nlevels", 1) > 1)
    for idx, row in frame.iterrows():
        as_of_date = idx.date() if hasattr(idx, "date") else _parse_iso_date(str(idx))

        if is_multi:
            for col in columns:
                ticker = str(col[0]).strip()
                field = str(col[1]).strip()
                raw_value = row[col]
                records.append(
                    BloombergRecord(
                        ticker=ticker,
                        field=field,
                        as_of_date=as_of_date,
                        value=_safe_float(raw_value),
                        currency=None,
                        source="bloomberg_blp",
                        raw_payload={"ticker": ticker, "field": field, "date": str(as_of_date), "value": raw_value},
                    )
                )
        else:
            for col in columns:
                col_name = str(col)
                if "|" in col_name:
                    ticker, field = [x.strip() for x in col_name.split("|", 1)]
                elif default_ticker and col_name in requested_fields:
                    # pdblp single-ticker bdh often returns plain field columns
                    ticker, field = default_ticker, col_name
                elif len(requested_fields) == 1:
                    ticker, field = col_name.strip(), requested_fields[0]
                else:
                    # flat columns with multi-fields are ambiguous; skip noisy parsing
                    continue
                raw_value = row[col]
                records.append(
                    BloombergRecord(
                        ticker=ticker,
                        field=field,
                        as_of_date=as_of_date,
                        value=_safe_float(raw_value),
                        currency=None,
                        source="bloomberg_blp",
                        raw_payload={"ticker": ticker, "field": field, "date": str(as_of_date), "value": raw_value},
                    )
                )
    return records


def fetch_bloomberg_data(
    tickers: list[str],
    fields: list[str],
    start_date: date,
    end_date: date,
) -> list[BloombergRecord]:
    """
    Fetch data from Bloomberg via the `pdblp` package.
    """
    try:
        import pdblp  # type: ignore
    except Exception as exc:
        raise RuntimeError("Cannot import `pdblp`. Please install it first: pip install pdblp") from exc

    start_raw = start_date.strftime("%Y%m%d")
    end_raw = end_date.strftime("%Y%m%d")

    # Align with DataHandler.py usage:
    # pdblp.BCon(debug=False, port=8194, timeout=500000) + bdh(..., elms=[])
    con = pdblp.BCon(debug=False, port=8194, timeout=500000)
    try:
        con.start()

        try:
            data = con.bdh(
                tickers=tickers,
                flds=fields,
                start_date=start_raw,
                end_date=end_raw,
                elms=[],
            )
            if hasattr(data, "iterrows") and hasattr(data, "columns"):
                return _records_from_bdh_frame(data, fields)
        except KeyError as exc:
            # pdblp can raise KeyError('securityData') on mixed valid/invalid securities.
            # Fall back to per-ticker requests to isolate bad tickers and keep good data.
            if "securityData" not in str(exc):
                raise
        except Exception:
            # Batch call failed for other reasons; try single-ticker fallback for better diagnostics.
            pass

        all_records: list[BloombergRecord] = []
        failed_tickers: list[str] = []
        failed_reasons: dict[str, str] = {}
        for ticker in tickers:
            try:
                ticker_data = con.bdh(
                    tickers=ticker,
                    flds=fields,
                    start_date=start_raw,
                    end_date=end_raw,
                    elms=[],
                )
                if hasattr(ticker_data, "iterrows") and hasattr(ticker_data, "columns"):
                    all_records.extend(_records_from_bdh_frame(ticker_data, fields, default_ticker=ticker))
                else:
                    failed_tickers.append(ticker)
                    failed_reasons[ticker] = f"unexpected_response_type={type(ticker_data).__name__}"
            except Exception as exc:
                failed_tickers.append(ticker)
                failed_reasons[ticker] = f"{type(exc).__name__}: {exc}"

        if not all_records:
            raise RuntimeError(
                "Bloomberg download failed for all tickers. "
                f"tickers={tickers}, fields={fields}, failed_tickers={failed_tickers}, "
                f"failed_reasons={failed_reasons}"
            )

        if failed_tickers:
            print(
                json.dumps(
                    {
                        "warning": "Some tickers failed in pdblp query.",
                        "failed_tickers": failed_tickers,
                        "failed_reasons": failed_reasons,
                    },
                    ensure_ascii=False,
                )
            )
        return all_records
    finally:
        # Ensure connection is always closed even when Bloomberg request fails.
        try:
            con.stop()
        except Exception:
            pass


def upsert_to_neon(records: list[BloombergRecord], neon_config_path: str = "neon_connection_string.json") -> int:
    """
    Upload price fields into Neon table `market_prices`.
    Dynamically expands table columns for requested fields.
    """
    database_url = _load_database_url(neon_config_path)
    conn = psycopg2.connect(database_url)
    try:
        if not records:
            return 0

        # Determine which fields we have and ensure columns exist.
        fields_present = sorted({r.field.strip().upper() for r in records if r.field})
        if not fields_present:
            return 0

        columns_to_ensure: dict[str, str] = {}
        for f in fields_present:
            col = _price_field_to_column(f)
            if col in {"ticker", "price_date", "updated_at"}:
                continue
            # Prices: numeric columns; allow NULL
            columns_to_ensure[col] = "NUMERIC(14, 4) NULL"

        if columns_to_ensure:
            _ensure_columns_exist(conn, "market_prices", columns_to_ensure)

        # Pivot into (ticker, date) rows with multiple columns
        col_order = [_price_field_to_column(f) for f in fields_present]
        col_order = [c for c in col_order if c not in {"ticker", "price_date", "updated_at"}]
        if not col_order:
            return 0

        pivot: dict[tuple[str, date], dict[str, float | None]] = {}
        for r in records:
            t = r.ticker
            d = r.as_of_date
            col = _price_field_to_column(r.field)
            if col not in col_order:
                continue
            pivot.setdefault((t, d), {})[col] = _safe_float(r.value)

        rows = []
        for (t, d), values in pivot.items():
            row = [t, d]
            for col in col_order:
                row.append(values.get(col))
            rows.append(tuple(row))

        if not rows:
            return 0

        insert_cols_sql = ", ".join(['"ticker"', '"price_date"'] + [f'"{c}"' for c in col_order] + ['"updated_at"'])
        conflict_set_sql = ", ".join([f'"{c}" = EXCLUDED."{c}"' for c in col_order] + ['"updated_at" = EXCLUDED."updated_at"'])
        template = "(" + ", ".join(["%s"] * (2 + len(col_order))) + ", NOW())"
        with conn.cursor() as cur:
            execute_values(
                cur,
                f"""
                INSERT INTO market_prices ({insert_cols_sql})
                VALUES %s
                ON CONFLICT (ticker, price_date) DO UPDATE SET
                    {conflict_set_sql}
                """,
                rows,
                template=template,
            )
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _fetch_macro_dataframe_pdblp(ticker: str, fields: list[str], start_date: date, end_date: date) -> Any:
    """
    Download macro series via pdblp, aligned with DataHandler.py:
      - pdblp.BCon(debug=False, port=8194, timeout=500000)
      - bdh(ticker, keyword, start, end, elms=[])
    """
    try:
        import pdblp  # type: ignore
    except Exception as exc:
        raise RuntimeError("Cannot import `pdblp`. Please install it first: pip install pdblp") from exc

    start_raw = start_date.strftime("%Y%m%d")
    end_raw = end_date.strftime("%Y%m%d")
    keyword = fields

    con = pdblp.BCon(debug=False, port=8194, timeout=500000)
    try:
        con.start()
        return con.bdh(ticker, keyword, start_raw, end_raw, elms=[])
    finally:
        try:
            con.stop()
        except Exception:
            pass


def _process_macro_bdh_output(data: Any, ticker: str, requested_fields: list[str]) -> list[dict[str, Any]]:
    """
    Convert pdblp bdh output into rows for `macro_data_revisions`.
    Following DataHandler.bbg_data_process behavior:
      - drop all-NA rows
      - use `data[ticker].reset_index()`
      - drop rows with NA PX_LAST
      - if ECO_RELEASE_DT exists: fillna with observation date then use it as `date`
      - de-duplicate by date, keep last
    """
    # Lazy import to avoid making pandas a hard dependency for other modes
    import pandas as pd  # type: ignore

    if not hasattr(data, "dropna"):
        raise RuntimeError(f"Unexpected pdblp output type for {ticker}: {type(data).__name__}")

    df = data.dropna(how="all")
    try:
        df = df[ticker].reset_index()
    except Exception as exc:
        raise RuntimeError(f"pdblp output missing expected ticker column: {ticker}") from exc

    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})

    if "PX_LAST" not in df.columns and "PX_LAST" in requested_fields:
        return []

    if "PX_LAST" in df.columns:
        df = df.dropna(subset=["PX_LAST"])

    # DataHandler: drop columns with >30% NA
    for col in list(df.columns):
        try:
            if df[col].isna().sum() / max(len(df), 1) > 0.3:
                df = df.drop(columns=[col])
        except Exception:
            continue

    # ECO_RELEASE_DT handling (DataHandler uses it to redefine `date`)
    if "ECO_RELEASE_DT" in df.columns and "ECO_RELEASE_DT" in requested_fields:
        # Preserve raw interface value for eco_release_dt column.
        # If the interface didn't return it (or returns 0/empty), we should store NULL,
        # not a derived/defaulted timestamp like epoch.
        df["__eco_release_dt_raw"] = df["ECO_RELEASE_DT"]

        df["date"] = pd.to_datetime(df["date"])
        df["ECO_RELEASE_DT"] = df["ECO_RELEASE_DT"].fillna(df["date"].dt.strftime("%Y%m%d"))
        df["date"] = pd.to_datetime(df["ECO_RELEASE_DT"].astype(int).astype(str))

    df = df.loc[~df["date"].duplicated(keep="last")]

    rows: list[dict[str, Any]] = []
    for _, r in df.iterrows():
        dt = pd.to_datetime(r["date"])
        data_date: date = dt.date()

        row_out: dict[str, Any] = {
            "ticker": ticker,
            "data_date": data_date,
        }

        if "ECO_RELEASE_DT" in requested_fields and "ECO_RELEASE_DT" in df.columns:
            # Only write the raw value as eco_release_dt; missing/0 should become NULL.
            row_out["eco_release_dt"] = _parse_optional_timestamptz(r.get("__eco_release_dt_raw"))

        for f in requested_fields:
            col = _macro_field_to_column(f)
            if col in {"ticker", "data_date"}:
                continue
            if col == "eco_release_dt":
                continue
            if f in df.columns:
                row_out[col] = _safe_float(r.get(f))
            else:
                # allow missing columns in response
                row_out[col] = None

        rows.append(row_out)
    return rows


def upsert_macro_revision(conn: Any, data: dict[str, Any], value_columns: list[str]) -> bool:
    """
    Revision upsert for one (ticker, data_date).
    Returns True when an INSERT occurs (Original or Revision), False when unchanged / skipped.
    """
    ticker = str(data["ticker"])
    data_date = data["data_date"]
    # If all numeric fields are None, skip
    has_any_value = any(data.get(c) is not None for c in value_columns if c != "eco_release_dt")
    if not has_any_value:
        return False

    with conn.cursor() as cur:
        select_cols = ", ".join([f'"{c}"' for c in value_columns] + ['revision_num'])
        cur.execute(
            f"""
            SELECT {select_cols}
            FROM macro_data_revisions
            WHERE ticker = %s AND data_date = %s AND is_latest = TRUE
            """,
            (ticker, data_date),
        )
        existing = cur.fetchone()

        if existing:
            old_values = dict(zip(value_columns + ["revision_num"], existing))
            old_rev_num = old_values["revision_num"]
            changed = False
            for c in value_columns:
                new_v = data.get(c)
                old_v = old_values.get(c)
                if c == "eco_release_dt":
                    if (old_v is None) != (new_v is None) or (old_v is not None and new_v is not None and str(old_v) != str(new_v)):
                        changed = True
                        break
                    continue
                try:
                    if new_v is None and old_v is None:
                        continue
                    if new_v is None or old_v is None:
                        changed = True
                        break
                    if float(old_v) != float(new_v):
                        changed = True
                        break
                except Exception:
                    if str(old_v) != str(new_v):
                        changed = True
                        break

            if not changed:
                return False

            cur.execute(
                """
                UPDATE macro_data_revisions
                SET is_latest = FALSE
                WHERE ticker = %s AND data_date = %s AND is_latest = TRUE
                """,
                (ticker, data_date),
            )
            insert_cols = ["ticker", "data_date"] + value_columns + ["revision_num", "is_latest", "data_status"]
            cols_sql = ", ".join([f'"{c}"' for c in insert_cols])
            placeholders = ", ".join(["%s"] * (2 + len(value_columns) + 1) + ["TRUE", "'Revision'"])
            values = [ticker, data_date] + [data.get(c) for c in value_columns] + [int(old_rev_num) + 1]
            cur.execute(
                f'INSERT INTO macro_data_revisions ({cols_sql}) VALUES ({placeholders})',
                tuple(values),
            )
            return True

        insert_cols = ["ticker", "data_date"] + value_columns + ["revision_num", "is_latest", "data_status"]
        cols_sql = ", ".join([f'"{c}"' for c in insert_cols])
        placeholders = ", ".join(["%s"] * (2 + len(value_columns) + 1) + ["TRUE", "'Original'"])
        values = [ticker, data_date] + [data.get(c) for c in value_columns] + [0]
        cur.execute(
            f'INSERT INTO macro_data_revisions ({cols_sql}) VALUES ({placeholders})',
            tuple(values),
        )
        return True


def update_macro_data_from_config(
    config_path: str = "bloomberg_download_config.json",
    neon_config_path: str = "neon_connection_string.json",
) -> dict[str, Any]:
    """
    Download macro data for tickers in config and upsert into `macro_data_revisions`.
    Uses config `start-date`, and sets end-date to today.
    """
    config = load_download_config(config_path)
    # Macro tickers are separated from price tickers in config.
    # Backward compatible: fall back to `tickers` if `macro_tickers` not provided.
    tickers_raw = config.get("macro_tickers")
    if tickers_raw is None:
        tickers_raw = config.get("tickers") or []
    start_date_raw = config.get("start-date")
    macro_fields_raw = config.get("macro_fields")
    if macro_fields_raw is None:
        # backward compatible
        macro_fields_raw = ["PX_LAST", "ECO_RELEASE_DT", "BN_SURVEY_MEDIAN"]
    if not isinstance(start_date_raw, str) or not start_date_raw.strip():
        raise RuntimeError("Missing `start-date` in config.")

    if isinstance(tickers_raw, str):
        tickers = [item.strip() for item in tickers_raw.split(",") if item.strip()]
    elif isinstance(tickers_raw, list):
        tickers = [str(item).strip() for item in tickers_raw if str(item).strip()]
    else:
        tickers = []
    if not tickers:
        raise RuntimeError("No valid tickers in config.")

    start_date = _parse_iso_date(start_date_raw.strip())
    today = date.today()

    database_url = _load_database_url(neon_config_path)
    conn = psycopg2.connect(database_url)
    try:
        if isinstance(macro_fields_raw, str):
            macro_fields = [x.strip() for x in macro_fields_raw.split(",") if x.strip()]
        elif isinstance(macro_fields_raw, list):
            macro_fields = [str(x).strip() for x in macro_fields_raw if str(x).strip()]
        else:
            macro_fields = []
        if not macro_fields:
            raise RuntimeError("No valid macro_fields in config.")

        # Ensure columns exist in macro table
        columns_to_ensure: dict[str, str] = {}
        value_columns: list[str] = []
        for f in macro_fields:
            col = _macro_field_to_column(f)
            if col in {"ticker", "data_date", "revision_num", "is_latest", "data_status", "created_at", "id"}:
                continue
            if col == "eco_release_dt":
                columns_to_ensure[col] = "TIMESTAMPTZ NULL"
            else:
                columns_to_ensure[col] = "NUMERIC(14, 4) NULL"
            value_columns.append(col)
        if columns_to_ensure:
            _ensure_columns_exist(conn, "macro_data_revisions", columns_to_ensure)

        total_downloaded = 0
        total_inserted = 0
        per_ticker: list[dict[str, Any]] = []

        for ticker in tickers:
            raw = _fetch_macro_dataframe_pdblp(ticker, fields=macro_fields, start_date=start_date, end_date=today)
            rows = _process_macro_bdh_output(raw, ticker=ticker, requested_fields=macro_fields)
            total_downloaded += len(rows)

            inserted = 0
            skipped = 0
            for row in rows:
                if upsert_macro_revision(conn, row, value_columns=value_columns):
                    inserted += 1
                else:
                    skipped += 1
            total_inserted += inserted

            per_ticker.append(
                {
                    "ticker": ticker,
                    "download_start": start_date.isoformat(),
                    "download_end": today.isoformat(),
                    "rows_downloaded": len(rows),
                    "rows_inserted": inserted,
                    "rows_skipped": skipped,
                }
            )

        conn.commit()
        return {
            "ok": True,
            "mode": "macro_revisions",
            "config": config_path,
            "neon_config": neon_config_path,
            "tickers": tickers,
            "start_date": start_date.isoformat(),
            "end_date": today.isoformat(),
            "rows_downloaded": total_downloaded,
            "rows_inserted": total_inserted,
            "per_ticker": per_ticker,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _get_latest_price_dates(tickers: list[str], neon_config_path: str) -> dict[str, date]:
    database_url = _load_database_url(neon_config_path)
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ticker, MAX(price_date) AS max_date
                FROM market_prices
                WHERE ticker = ANY(%s)
                GROUP BY ticker
                """,
                (tickers,),
            )
            rows = cur.fetchall()
        return {str(row[0]): row[1] for row in rows if row and row[1] is not None}
    finally:
        conn.close()


def delete_market_data_by_tickers(tickers: list[str], neon_config_path: str = "neon_connection_string.json") -> dict[str, Any]:
    """
    Delete all historical rows for the given tickers from `market_prices`.
    """
    normalized_tickers = [t.strip() for t in tickers if t and t.strip()]
    if not normalized_tickers:
        raise RuntimeError("No valid tickers were provided for deletion.")

    database_url = _load_database_url(neon_config_path)
    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM market_prices
                WHERE ticker = ANY(%s)
                """,
                (normalized_tickers,),
            )
            deleted_rows = cur.rowcount
        conn.commit()
        return {
            "ok": True,
            "action": "delete_market_data",
            "tickers": normalized_tickers,
            "deleted_rows": int(deleted_rows),
            "neon_config": neon_config_path,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_predefined_tickers(neon_config_path: str = "neon_connection_string.json") -> dict[str, Any]:
    """
    Delete ticker history using a predefined list.
    Edit `tickers_to_delete` only when you intentionally want to run deletion.
    """
    tickers_to_delete = [
        # "AAPL US Equity",
        # "MSFT US Equity",
    ]
    return delete_market_data_by_tickers(tickers=tickers_to_delete, neon_config_path=neon_config_path)


def update_market_data_from_config(
    config_path: str = "bloomberg_download_config.json",
    neon_config_path: str = "neon_connection_string.json",
    fetch_only: bool = False,
) -> dict[str, Any]:
    """
    Incremental update based on config tickers/fields:
    - If ticker exists in DB, download from latest price_date (inclusive) to today.
    - If ticker not in DB, download from config start-date to today.
    """
    config = load_download_config(config_path)
    # Price tickers live under `tickers` in config (separate from macro_tickers).
    tickers_raw = config.get("tickers") or []
    # Dynamic price fields (backward compatible: fall back to `fields`)
    fields_raw = config.get("price_fields")
    if fields_raw is None:
        fields_raw = config.get("fields") or []
    start_date_raw = config.get("start-date")

    if isinstance(tickers_raw, str):
        tickers = [item.strip() for item in tickers_raw.split(",") if item.strip()]
    elif isinstance(tickers_raw, list):
        tickers = [str(item).strip() for item in tickers_raw if str(item).strip()]
    else:
        tickers = []

    if isinstance(fields_raw, str):
        fields = [item.strip() for item in fields_raw.split(",") if item.strip()]
    elif isinstance(fields_raw, list):
        fields = [str(item).strip() for item in fields_raw if str(item).strip()]
    else:
        fields = []

    if not isinstance(start_date_raw, str) or not start_date_raw.strip():
        raise RuntimeError("Missing `start-date` in config.")
    if not tickers:
        raise RuntimeError("No valid tickers in config.")
    if not fields:
        raise RuntimeError("No valid fields in config.")

    global_start_date = _parse_iso_date(start_date_raw.strip())
    today = date.today()
    latest_dates = _get_latest_price_dates(tickers, neon_config_path=neon_config_path)

    all_records: list[BloombergRecord] = []
    windows: list[dict[str, str]] = []

    for ticker in tickers:
        ticker_start = latest_dates.get(ticker, global_start_date)
        ticker_end = today
        if ticker_start > ticker_end:
            windows.append(
                {
                    "ticker": ticker,
                    "start_date": ticker_start.isoformat(),
                    "end_date": ticker_end.isoformat(),
                    "mode": "skip_future_start_date",
                }
            )
            continue

        ticker_records = fetch_bloomberg_data(
            tickers=[ticker],
            fields=fields,
            start_date=ticker_start,
            end_date=ticker_end,
        )
        all_records.extend(ticker_records)
        windows.append(
            {
                "ticker": ticker,
                "start_date": ticker_start.isoformat(),
                "end_date": ticker_end.isoformat(),
                "mode": "incremental" if ticker in latest_dates else "full",
            }
        )

    upserted = 0
    upload_skipped = False
    if fetch_only:
        upload_skipped = True
    else:
        upserted = upsert_to_neon(all_records, neon_config_path=neon_config_path)

    return {
        "ok": True,
        "config": config_path,
        "neon_config": neon_config_path,
        "tickers": tickers,
        "fields": fields,
        "today": today.isoformat(),
        "windows": windows,
        "fetched_records": len(all_records),
        "upserted_records": upserted,
        "upload_skipped": upload_skipped,
        "uploaded_field": "PX_LAST",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Bloomberg data and upload to Neon.")
    parser.add_argument(
        "--config",
        default="bloomberg_download_config.json",
        help="Path to JSON config file.",
    )
    parser.add_argument(
        "--neon-config",
        default="neon_connection_string.json",
        help="Path to Neon JSON containing `connection_string`.",
    )
    parser.add_argument(
        "--tickers",
        required=False,
        help="Comma-separated tickers. Example: 'AAPL US Equity,MSFT US Equity'",
    )
    parser.add_argument(
        "--fields",
        required=False,
        help="Comma-separated fields. Example: 'PX_LAST,PX_VOLUME'",
    )
    parser.add_argument("--start-date", required=False, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=False, help="YYYY-MM-DD")
    parser.add_argument(
        "--fetch-only",
        action="store_true",
        help="Only download Bloomberg data; skip Neon upload.",
    )
    parser.add_argument(
        "--macro",
        action="store_true",
        help="Run macro download + revision upsert into `macro_data_revisions`.",
    )
    return parser.parse_args()


def main() -> None:
    
    result = update_macro_data_from_config(
            config_path="bloomberg_download_config.json",
            neon_config_path="neon_connection_string.json",
        )        
    print(json.dumps(result, ensure_ascii=False))
    return

    # result = update_market_data_from_config(
    #     config_path=args.config,
    #     neon_config_path=args.neon_config,
    #     fetch_only=args.fetch_only,
    # )
    # print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()