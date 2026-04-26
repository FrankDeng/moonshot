import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Iterable

import requests
from requests_oauthlib import OAuth1


@dataclass
class AnalystResult:
    indicator_name: str
    workflow_name: str
    description: str
    summary: str
    generated_at: str | None
    figure_path: str | None
    analysis_path: str | None


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _load_x_credentials(config_path: str = "X_connection_string.json") -> dict[str, str]:
    """
    Load X credentials with priority:
    1) Environment variables
    2) Local JSON config file (for local runs)
    """
    creds = {
        "api_key": os.environ.get("X_API_KEY", "").strip(),
        "api_secret": os.environ.get("X_API_SECRET", "").strip(),
        "access_token": os.environ.get("X_ACCESS_TOKEN", "").strip(),
        "access_token_secret": os.environ.get("X_ACCESS_TOKEN_SECRET", "").strip(),
    }
    if all(creds.values()):
        return creds

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as fp:
            loaded = json.load(fp)
        if isinstance(loaded, dict):
            creds["api_key"] = creds["api_key"] or _safe_text(loaded.get("X_API_KEY") or loaded.get("api_key"))
            creds["api_secret"] = creds["api_secret"] or _safe_text(loaded.get("X_API_SECRET") or loaded.get("api_secret"))
            creds["access_token"] = creds["access_token"] or _safe_text(
                loaded.get("X_ACCESS_TOKEN") or loaded.get("access_token")
            )
            creds["access_token_secret"] = creds["access_token_secret"] or _safe_text(
                loaded.get("X_ACCESS_TOKEN_SECRET") or loaded.get("access_token_secret")
            )

    missing = [k for k, v in creds.items() if not v]
    if missing:
        raise RuntimeError(
            "Missing X credentials fields: "
            f"{missing}. Provide env vars X_API_KEY/X_API_SECRET/X_ACCESS_TOKEN/X_ACCESS_TOKEN_SECRET "
            f"or fill them in {config_path}."
        )
    return creds


class Publisher:
    channel_name: str = "base"

    def publish(self, result: AnalystResult, dry_run: bool = False) -> dict[str, Any]:
        raise NotImplementedError


class XPublisher(Publisher):
    channel_name = "x"

    def __init__(self) -> None:
        creds = _load_x_credentials()
        self.api_key = creds["api_key"]
        self.api_secret = creds["api_secret"]
        self.access_token = creds["access_token"]
        self.access_token_secret = creds["access_token_secret"]

        self.auth = OAuth1(
            client_key=self.api_key,
            client_secret=self.api_secret,
            resource_owner_key=self.access_token,
            resource_owner_secret=self.access_token_secret,
            signature_type="auth_header",
        )

    @staticmethod
    def _truncate_text(text: str, limit: int = 280) -> str:
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    def _compose_text(self, result: AnalystResult) -> str:
        title = f"[{result.indicator_name}]"
        desc = result.description or "Macro cycle update"
        summary = result.summary or "No LLM summary available."
        # Keep concise for X; detailed full text remains in local artifacts/DB.
        body = f"{title}\n{desc}\n\n{summary}"
        return self._truncate_text(body, 280)

    def _upload_media_if_exists(self, file_path: str | None) -> str | None:
        if not file_path:
            return None
        if not os.path.exists(file_path):
            return None
        # X media upload API (v1.1) for images.
        upload_url = "https://upload.twitter.com/1.1/media/upload.json"
        with open(file_path, "rb") as fp:
            files = {"media": fp}
            resp = requests.post(upload_url, auth=self.auth, files=files, timeout=60)
        if resp.status_code >= 300:
            raise RuntimeError(f"X media upload failed {resp.status_code}: {resp.text[:300]}")
        body = resp.json()
        media_id = body.get("media_id_string")
        return str(media_id) if media_id else None

    def publish(self, result: AnalystResult, dry_run: bool = False) -> dict[str, Any]:
        text = self._compose_text(result)
        if dry_run:
            return {
                "ok": True,
                "channel": self.channel_name,
                "dry_run": True,
                "text_preview": text,
                "indicator_name": result.indicator_name,
            }

        media_id = None
        # Optional: only if figure path is image (png/jpg). HTML won't be uploaded.
        if result.figure_path and result.figure_path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            media_id = self._upload_media_if_exists(result.figure_path)

        # Prefer v2, fallback to v1.1 on permission/tier issues.
        post_url_v2 = "https://api.twitter.com/2/tweets"
        payload_v2: dict[str, Any] = {"text": text}
        if media_id:
            payload_v2["media"] = {"media_ids": [media_id]}

        resp = requests.post(post_url_v2, auth=self.auth, json=payload_v2, timeout=60)
        if resp.status_code < 300:
            body = resp.json()
            tweet_id = ((body.get("data") or {}).get("id")) if isinstance(body, dict) else None
            return {
                "ok": True,
                "channel": self.channel_name,
                "indicator_name": result.indicator_name,
                "tweet_id": tweet_id,
                "endpoint": "v2",
            }

        # Collect request id for debugging (useful with X support)
        req_id = resp.headers.get("x-request-id") or resp.headers.get("X-Request-Id")
        # Fallback cases: 401/403 often indicate tier or oauth permission mismatch for v2.
        if resp.status_code in (401, 403):
            post_url_v11 = "https://api.twitter.com/1.1/statuses/update.json"
            payload_v11 = {"status": text}
            resp11 = requests.post(post_url_v11, auth=self.auth, data=payload_v11, timeout=60)
            if resp11.status_code < 300:
                body11 = resp11.json()
                tweet_id = body11.get("id_str") or body11.get("id")
                return {
                    "ok": True,
                    "channel": self.channel_name,
                    "indicator_name": result.indicator_name,
                    "tweet_id": str(tweet_id) if tweet_id else None,
                    "endpoint": "v1.1",
                    "v2_error_status": resp.status_code,
                    "v2_request_id": req_id,
                }
            req_id11 = resp11.headers.get("x-request-id") or resp11.headers.get("X-Request-Id")
            raise RuntimeError(
                f"X publish failed. v2_status={resp.status_code}, v2_request_id={req_id}, "
                f"v2_body={resp.text[:300]} | "
                f"v1.1_status={resp11.status_code}, v1.1_request_id={req_id11}, v1.1_body={resp11.text[:300]}"
            )

        raise RuntimeError(f"X publish failed {resp.status_code} (request_id={req_id}): {resp.text[:300]}")


def build_publishers(channels: Iterable[str]) -> list[Publisher]:
    publishers: list[Publisher] = []
    for c in channels:
        key = c.strip().lower()
        if key == "x":
            publishers.append(XPublisher())
            continue
        raise RuntimeError(f"Unsupported channel: {c}")
    return publishers


def publish_single_result(
    channels: list[str],
    result: AnalystResult,
    dry_run: bool = False,
) -> dict[str, Any]:
    publishers = build_publishers(channels)
    publish_logs: list[dict[str, Any]] = []
    for p in publishers:
        log = p.publish(result, dry_run=dry_run)
        publish_logs.append(log)
    return {
        "ok": True,
        "channels": channels,
        "result_count": 1,
        "published": publish_logs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish one analyst result to channels.")
    parser.add_argument("--channels", default="x", help="Comma-separated channels, e.g. x")
    parser.add_argument("--indicator-name", required=True, help="Indicator name.")
    parser.add_argument("--workflow-name", default="macro-workflow", help="Workflow name.")
    parser.add_argument("--description", default="", help="Short description.")
    parser.add_argument("--summary", default="", help="LLM summary content.")
    parser.add_argument("--generated-at", default="", help="Generated timestamp text.")
    parser.add_argument("--figure-path", default="", help="Optional local figure path (image preferred).")
    parser.add_argument("--analysis-path", default="", help="Optional local analysis file path.")
    parser.add_argument("--dry-run", action="store_true", help="Preview text without posting.")
    args = parser.parse_args()

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    if not channels:
        raise RuntimeError("No channels provided.")

    result = AnalystResult(
        indicator_name=_safe_text(args.indicator_name),
        workflow_name=_safe_text(args.workflow_name),
        description=_safe_text(args.description),
        summary=_safe_text(args.summary),
        generated_at=_safe_text(args.generated_at) or None,
        figure_path=_safe_text(args.figure_path) or None,
        analysis_path=_safe_text(args.analysis_path) or None,
    )
    out = publish_single_result(channels=channels, result=result, dry_run=args.dry_run)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
