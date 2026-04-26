import json
import os
import re
import math
import hashlib
from datetime import datetime
from urllib.parse import parse_qs
from typing import Any

import modal
from fastapi import HTTPException, Request

app = modal.App("folo-article-pipeline")

# 固定 embedding 配置，确保向量空间稳定
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
EMBEDDING_MODEL_VERSION = f"{EMBEDDING_MODEL_NAME}:{EMBEDDING_DIMENSION}"
METADATA_MODEL_NAME = "deepseek-chat"
METADATA_SCHEMA_VERSION = "v2"
PIPELINE_VERSION = "folo-pipeline-v1"
ANALYSIS_MODEL_VERSION = f"{PIPELINE_VERSION}|emb:{EMBEDDING_MODEL_VERSION}|meta:{METADATA_MODEL_NAME}|schema:{METADATA_SCHEMA_VERSION}"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.115.6",
        "pydantic==2.10.4",
        "requests==2.32.3",
        "sentence-transformers==3.0.1",
        "torch==2.6.0",
        "langchain-text-splitters==0.3.2",
        "psycopg2-binary==2.9.10",
    )
    .run_commands(
        "python3 -c \"from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')\""
    )
)

# 持久化保存处理结果，便于后续查询或回放
results_store = modal.Dict.from_name("folo-article-results", create_if_missing=True)

# 记录 raw_ingestion_log 的清理日期：确保每天最多清理一次
cleanup_store = modal.Dict.from_name("folo-raw-ingestion-cleanup", create_if_missing=True)


def normalize_whitespace(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    # 只压缩行内空白，不破坏换行结构
    text = re.sub(r"[ \t]+", " ", text)
    # 清理每行首尾空白，保留正文分段
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # 最多保留一个空行分段
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_text_for_chunking(text: str) -> str:
    """
    在切分前清理常见噪音段落，避免污染向量空间。
    """
    if not text:
        return ""

    # 先把常见块级标签替换为换行，尽量保留段落结构
    plain = re.sub(r"(?is)</?(p|div|h[1-6]|li|ul|ol|br)[^>]*>", "\n", text)
    # 删除图片标签本身
    plain = re.sub(r"(?is)<img[^>]*>", " ", plain)
    # 链接保留可见文本，去掉 href 等属性
    plain = re.sub(r"(?is)<a[^>]*>(.*?)</a>", r"\1", plain)
    # 清理剩余 HTML 标签
    plain = re.sub(r"(?is)<[^>]+>", " ", plain)

    # 清理占位符和模板变量
    plain = re.sub(r"\[(?:title|url|content_html|content_markdown|summary)\]", " ", plain, flags=re.I)
    plain = re.sub(r"\{\{[^}]+\}\}", " ", plain)

    # 仅替换 URL，不删除整行
    plain = re.sub(r"(?i)(?:http|https)://\S+", " ", plain)
    plain = re.sub(r"(?i)\bwww\.\S+", " ", plain)
    plain = plain.replace("\xa0", " ").replace("&nbsp;", " ")
    plain = plain.replace("&#x26;", "&")
    plain = normalize_whitespace(plain)

    # 只删除“纯噪音行”，避免误删正文
    noise_line_patterns = [
        r"(?i)^免责声明[:：]?\s*$",
        r"(?i)^风险提示[:：]?\s*$",
        r"(?i)^投资建议[:：]?\s*$",
        r"(?i)^广告[:：]?\s*$",
        r"(?i)^推广[:：]?\s*$",
        r"(?i)^版权所有.*$",
        r"^\s*第?\s*\d+\s*/\s*\d+\s*页\s*$",
        r"^\s*第?\s*\d+\s*页\s*$",
        r"^[\W_]+$",
    ]

    lines = [line.strip() for line in plain.split("\n") if line.strip()]
    cleaned_lines: list[str] = []
    for line in lines:
        # 过滤过短纯噪音行
        if len(line) <= 2:
            continue

        is_noise = any(re.fullmatch(pattern, line) for pattern in noise_line_patterns)
        if is_noise:
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    # 合并重复段落（研报页脚/版权文案常重复）
    deduped: list[str] = []
    seen: set[str] = set()
    for paragraph in [p.strip() for p in cleaned.split("\n") if p.strip()]:
        if paragraph in seen and len(paragraph) > 30:
            continue
        seen.add(paragraph)
        deduped.append(paragraph)

    cleaned_text = normalize_whitespace("\n".join(deduped))
    # 去除低信息密度符号串
    cleaned_text = re.sub(r"[`~^*_=\-]{2,}", " ", cleaned_text)
    cleaned_text = re.sub(r"[|<>]{2,}", " ", cleaned_text)
    return normalize_whitespace(cleaned_text)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    clean = normalize_whitespace(text)
    if not clean:
        return []
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    return [c.strip() for c in splitter.split_text(clean) if c and c.strip()]

def parse_webhook_payload(content_type: str, raw_body: str) -> dict[str, Any]:
    if "application/json" in content_type:
        try:
            loaded = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Invalid JSON payload.",
                    "hint": "Please use standard double quotes.",
                    "example": {"title": "...", "content": "..."},
                    "content_type": content_type,
                    "raw_body_preview": raw_body[:300],
                },
            ) from exc
        if not isinstance(loaded, dict):
            raise HTTPException(
                status_code=400,
                detail={"message": "Payload must be a JSON object.", "content_type": content_type},
            )
        return loaded
    elif "application/x-www-form-urlencoded" in content_type:
        form_data = parse_qs(raw_body, keep_blank_values=True)
        return {k: (v[-1] if len(v) == 1 else v) for k, v in form_data.items()}
    elif "text/plain" in content_type:
        # 一些 webhook 直接发原始文本，作为 content 兜底
        return {"content": raw_body}
    else:
        # 尝试按 JSON 解析；失败则降级为纯文本正文
        try:
            loaded = json.loads(raw_body)
            return loaded if isinstance(loaded, dict) else {"content": raw_body}
        except json.JSONDecodeError:
            return {"content": raw_body}

def parse_json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    # 优先直接按 JSON 解析
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    # 兼容模型返回 ```json ... ``` 的场景
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.S)
    if match:
        try:
            loaded = json.loads(match.group(1))
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass

    return {"raw": text}


def stable_hash(value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_content_hash(article_data: dict[str, Any], cleaned_content: str | None = None) -> str:
    if cleaned_content is None:
        source_content = str(article_data.get("content") or "")
        cleaned = clean_text_for_chunking(source_content)
        if not cleaned:
            cleaned = normalize_whitespace(source_content)
    else:
        cleaned = str(cleaned_content)
    return hashlib.sha256(cleaned.encode("utf-8")).hexdigest()


def parse_sentiment_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        raw = str(value).strip()
        if not raw or raw.upper() == "NA":
            return None
        try:
            score = float(raw)
        except ValueError:
            return None
    # 统一约束到 [-1, 1]，便于后续做排序和筛选
    if score < -1.0:
        return -1.0
    if score > 1.0:
        return 1.0
    return score


def _normalize_plain_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"有", "无", "{}", "[]", "none", "null", "nil"}:
        return None
    return text


def _normalize_metadata_text(value: Any) -> str:
    text = _normalize_plain_text(value)
    if text is None:
        return "NA"
    if text.strip().lower() in {"na", "n/a"}:
        return "NA"
    return text


def _normalize_metadata_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    normalized_items: list[str] = []
    for item in value:
        text = _normalize_plain_text(item)
        if text is None:
            continue
        lowered = text.lower()
        if lowered in {"na", "n/a"}:
            continue
        normalized_items.append(text)
    return normalized_items


def _normalize_article_scores(value: Any) -> dict[str, Any]:
    score_keys = [
        "content_innovation",
        "logic_rigor",
        "citation_evidence",
        "conclusion_accuracy",
        "overall_score",
    ]
    raw_scores = value if isinstance(value, dict) else {}
    normalized_scores: dict[str, Any] = {}
    for key in score_keys:
        raw = raw_scores.get(key)
        if isinstance(raw, (int, float)):
            candidate = int(raw)
            normalized_scores[key] = candidate if 1 <= candidate <= 10 else "NA"
            continue
        text = _normalize_plain_text(raw)
        if text is None:
            normalized_scores[key] = "NA"
            continue
        try:
            candidate = int(float(text))
            normalized_scores[key] = candidate if 1 <= candidate <= 10 else "NA"
        except ValueError:
            normalized_scores[key] = "NA"
    normalized_scores["score_reason"] = _normalize_metadata_text(raw_scores.get("score_reason"))
    return normalized_scores


def normalize_metadata_output(raw_metadata: Any) -> dict[str, Any]:
    metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
    sentiment = parse_sentiment_score(metadata.get("sentiment_score"))
    reading_time_minutes = metadata.get("reading_time_minutes")
    if not isinstance(reading_time_minutes, (int, float)):
        reading_time_minutes = None
    return {
        "summary": _normalize_metadata_text(metadata.get("summary")),
        "language": _normalize_metadata_text(metadata.get("language")),
        "categories": _normalize_metadata_list(metadata.get("categories")),
        "tags": _normalize_metadata_list(metadata.get("tags")),
        "entities": _normalize_metadata_list(metadata.get("entities")),
        "researched_ticker": _normalize_metadata_list(metadata.get("researched_ticker")),
        "industry": _normalize_metadata_list(metadata.get("industry")),
        "sentiment_score": "NA" if sentiment is None else sentiment,
        "reading_time_minutes": int(reading_time_minutes) if isinstance(reading_time_minutes, (int, float)) else None,
        "long_short_view": _normalize_metadata_text(metadata.get("long_short_view")),
        "target_timeframe": _normalize_metadata_text(metadata.get("target_timeframe")),
        "target_price": _normalize_metadata_text(metadata.get("target_price")),
        "core_logic": _normalize_metadata_text(metadata.get("core_logic")),
        "confidence_index": _normalize_metadata_text(metadata.get("confidence_index")),
        "article_scores": _normalize_article_scores(metadata.get("article_scores")),
    }


def _normalize_json_container(value: Any) -> Any:
    if isinstance(value, dict):
        return value if value else None
    if isinstance(value, list):
        return value if value else None
    return None


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return -1.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return -1.0
    return dot / (norm_a * norm_b)


def build_metadata_sections_from_chunks(
    cleaned_content: str,
    chunk_records: list[dict[str, Any]],
    long_article_threshold: int = 2200,
    min_sections_for_long: int = 2,
    max_sections_for_long: int = 3,
) -> list[str]:
    cleaned_content = normalize_whitespace(cleaned_content)
    if not cleaned_content:
        return []

    if not chunk_records:
        return [cleaned_content]

    total_len = len(cleaned_content)
    if total_len <= long_article_threshold:
        return [cleaned_content]

    target_sections = max_sections_for_long if len(chunk_records) >= 8 else min_sections_for_long
    target_sections = max(min_sections_for_long, min(max_sections_for_long, target_sections))
    target_sections = min(target_sections, len(chunk_records))

    groups: list[list[dict[str, Any]]] = [[c] for c in chunk_records]
    while len(groups) > target_sections:
        best_idx = 0
        best_sim = -2.0
        for i in range(len(groups) - 1):
            left_emb = groups[i][-1].get("embedding", [])
            right_emb = groups[i + 1][0].get("embedding", [])
            sim = cosine_similarity(left_emb, right_emb)
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        groups[best_idx] = groups[best_idx] + groups[best_idx + 1]
        del groups[best_idx + 1]

    sections: list[str] = []
    for group in groups:
        section_text = "\n\n".join(str(c.get("text", "")).strip() for c in group if str(c.get("text", "")).strip())
        if section_text:
            sections.append(section_text)
    return sections


def coerce_payload(raw_payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(raw_payload)

    # 兼容常见嵌套结构：{"data": {...}} / {"entry": {...}} / {"article": {...}} / {"post": {...}}
    for nested_key in ("data", "entry", "article", "post", "payload"):
        nested = payload.get(nested_key)
        if isinstance(nested, dict):
            payload = dict(nested)
            break

    # folo.is 或第三方转发可能把 title 包成数组
    title = payload.get("title")
    if isinstance(title, list):
        payload["title"] = " ".join(str(item) for item in title if item is not None).strip()

    # 兼容 folo 字段命名
    if not payload.get("published_at"):
        payload["published_at"] = payload.get("publishedAt") or payload.get("insertedAt")

    # 原始 webhook 里 feed.title 作为 source_title 透传
    if not payload.get("source_title"):
        payload["source_title"] = extract_feed_title_from_raw_payload(raw_payload)

    # 兼容常见正文字段，统一映射到 content
    content_candidates = [
        payload.get("content"),
        payload.get("body"),
        payload.get("text"),
        payload.get("article"),
    ]
    if not payload.get("content"):
        for candidate in content_candidates:
            if isinstance(candidate, str) and candidate.strip():
                payload["content"] = candidate
                break

    # 如果仍缺正文，联调阶段退化为使用标题作为最小内容，避免 webhook 一直 400
    if not payload.get("content"):
        fallback_title = payload.get("title")
        if isinstance(fallback_title, str) and fallback_title.strip():
            payload["content"] = fallback_title.strip()

    return payload


def extract_feed_title_from_raw_payload(raw_payload: dict[str, Any] | None) -> str | None:
    if not isinstance(raw_payload, dict):
        return None

    candidate_nodes: list[dict[str, Any]] = [raw_payload]
    for nested_key in ("data", "entry", "article", "post", "payload"):
        nested = raw_payload.get(nested_key)
        if isinstance(nested, dict):
            candidate_nodes.append(nested)

    for node in candidate_nodes:
        feed = node.get("feed")
        if isinstance(feed, dict):
            feed_title = feed.get("title")
            if feed_title is not None:
                feed_title_str = str(feed_title).strip()
                if feed_title_str:
                    return feed_title_str
    return None


@app.cls(
    image=image,
    secrets=[
        modal.Secret.from_name("deepseek-api-key"),
    ],
    gpu="T4",
    cpu=2.0,
    memory=4096,
)
class ArticlePipeline:
    @modal.enter()
    def load_models(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _extract_metadata_and_tags(self, sections: list[str]) -> dict[str, Any]:
        import requests

        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY")

        if not sections:
            return {
                "summary": "NA",
                "tags": [],
                "categories": [],
                "entities": [],
                "researched_ticker": [],
                "industry": [],
                "sentiment_score": "NA",
                "article_scores": {
                    "content_innovation": "NA",
                    "logic_rigor": "NA",
                    "citation_evidence": "NA",
                    "conclusion_accuracy": "NA",
                    "overall_score": "NA",
                    "score_reason": "NA",
                },
            }

        base_json_schema = (
            '{\n'
            '  "summary": "对整篇文章的极简中文摘要（80-150字）",\n'
            '  "language": "zh/en/other",\n'
            '  "categories": ["一级分类", "二级分类"],\n'
            '  "tags": ["标签1", "标签2"],\n'
            '  "entities": ["核心实体"],\n'
            '  "researched_ticker": ["文中提到的交易标的，如AAPL、TSLA、BTC、510300/NA"],\n'
            '  "industry": ["文章涉及的行业，如半导体、AI基础设施、消费电子/NA"],\n'
            '  "sentiment_score": "整体情绪分，-1到1之间的小数（看空为负，看多为正，中性接近0）/NA",\n'
            '  "reading_time_minutes": 5,\n'
            '  "long_short_view": "long/short/neutral/NA",\n'
            '  "target_timeframe": "如3个月/NA",\n'
            '  "target_price": "如XX美元/NA",\n'
            '  "core_logic": "作者分析理由/逻辑/NA",\n'
            '  "confidence_index": "1-10间的信心指数/NA",\n'
            '  "article_scores": {\n'
            '    "content_innovation": "内容创新评分，1-10整数/NA",\n'
            '    "logic_rigor": "逻辑严谨评分，1-10整数/NA",\n'
            '    "citation_evidence": "引用详实评分，1-10整数/NA",\n'
            '    "conclusion_accuracy": "结论准确评分，1-10整数/NA",\n'
            '    "overall_score": "综合评分，1-10整数/NA",\n'
            '    "score_reason": "评分理由，40-120字"\n'
            '  }\n'
            '}'
        )

        if len(sections) == 1:
            final_prompt = (
                "你是一个金融内容结构化分析助手。请根据给定文章内容，"
                "严格输出 JSON，无法判断的字段请写 NA。\n"
                "禁止输出“有”“无”“None”“{}”“[]”等占位词或符号；"
                "如果本来会输出这些值，统一改为 NA（必须是英文大写 NA）。\n"
                "其中 article_scores 必须给出评分：内容创新、逻辑严谨、引用详实、结论准确，"
                "评分范围为 1-10（整数）。\n"
                "请额外输出 researched_ticker、industry 和 sentiment_score；"
                "其中 sentiment_score 必须是 -1 到 1 之间的小数，无法判断时写 NA。\n"
                f"输出格式:\n{base_json_schema}\n"
                "不要输出其它解释。"
            )
            final_input = f"正文:\n{sections[0][:10000]}"
        else:
            chunk_summaries: list[str] = []
            for idx, section in enumerate(sections, start=1):
                chunk_prompt = (
                    "请对以下文章片段提炼核心观点（投资方向、逻辑、建议、风险），"
                    "返回 3-5 条要点，最多 120 字：\n\n"
                    f"{section[:4000]}"
                )
                chunk_body = {
                    "model": "deepseek-chat",
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": chunk_prompt}],
                }
                chunk_response = requests.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=chunk_body,
                    timeout=60,
                )
                chunk_response.raise_for_status()
                chunk_data = chunk_response.json()
                chunk_summary = chunk_data["choices"][0]["message"]["content"].strip()
                chunk_summaries.append(f"第{idx}段总结: {chunk_summary}")

            integrated_summary = "\n".join(chunk_summaries)
            final_prompt = (
                "你是一个金融内容结构化分析助手。\n"
                "现在给定文章分段总结，请综合后严格输出 JSON，无法判断写 NA。\n"
                "禁止输出“有”“无”“None”“{}”“[]”等占位词或符号；"
                "如果本来会输出这些值，统一改为 NA（必须是英文大写 NA）。\n"
                "其中 article_scores 必须给出评分：内容创新、逻辑严谨、引用详实、结论准确，"
                "评分范围为 1-10（整数）。\n"
                "请额外输出 researched_ticker、industry 和 sentiment_score；"
                "其中 sentiment_score 必须是 -1 到 1 之间的小数，无法判断时写 NA。\n"
                f"输出格式:\n{base_json_schema}\n"
                "不要输出其它解释。"
            )
            final_input = f"分段总结:\n{integrated_summary}"

        final_body = {
            "model": "deepseek-chat",
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": final_prompt},
                {"role": "user", "content": final_input},
            ],
        }

        response = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=final_body,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        output_text = data["choices"][0]["message"]["content"]
        return normalize_metadata_output(parse_json_from_text(output_text))

    @modal.method()
    def run(self, article: dict[str, Any]) -> dict[str, Any]:
        
        article_id = article.get("id") or article.get("url") or f"article-{int(datetime.utcnow().timestamp())}"
        title = article.get("title")
        url = article.get("url")
        content = article.get("content")
        author = article.get("author")
        author_url = article.get("authorUrl")
        inserted_at = article.get("insertedAt")
        published_at = article.get("publishedAt")
        media = article.get("media")
        categories = article.get("categories")
        attachments = article.get("attachments")
        extra = article.get("extra")
        language = article.get("language")
        summary = article.get("summary")
        source_title = article.get("source_title")
        if not content:
            raise ValueError("article.content is required")

        # print(content)
        cleaned_content = clean_text_for_chunking(content)
        if not cleaned_content:
            cleaned_content = normalize_whitespace(content)

        base_chunks = chunk_text(cleaned_content, chunk_size=800, overlap=120)
        if not base_chunks:
            raise ValueError("No valid chunks after cleaning")

        effective_date = article.get("published_at") or published_at or inserted_at or ""
        chunk_date = str(effective_date)[:10] if effective_date else "unknown"
        chunk_title = (title or "untitled").strip() if isinstance(title, str) else "untitled"
        chunk_author = (author or "unknown").strip() if isinstance(author, str) else "unknown"
        chunk_summary = (summary or "unknown").strip() if isinstance(summary, str) else "unknown"

        contextual_chunks = [
            f"Title: {chunk_title}\nDate: {chunk_date}\nAuthor: {chunk_author}\nSummary: {chunk_summary}\n{chunk}"
            for chunk in base_chunks
        ]
        vectors = self.embedding_model.encode(
            contextual_chunks,
            batch_size=12,
            normalize_embeddings=True,
        )
        chunk_records = []
        for idx, (chunk, emb) in enumerate(zip(base_chunks, vectors), start=1):
            if len(emb) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(emb)}"
                )
            chunk_records.append(
                {
                    "chunk_index": idx,
                    "text": chunk,
                    "summary": chunk_summary,
                    "author": chunk_author,
                    "date": chunk_date,
                    "title": chunk_title,
                    "embedding": emb.tolist(),
                    "embedding_dim": EMBEDDING_DIMENSION,
                }
            )
        

        metadata_sections = build_metadata_sections_from_chunks(
            cleaned_content,
            chunk_records,
            long_article_threshold=2200,
            min_sections_for_long=2,
            max_sections_for_long=4,
        )
        metadata = self._extract_metadata_and_tags(metadata_sections)
   
        result = {
            "article_id": article_id,
            "source": "folo.is",
            "source_title": source_title,
            "title": title,
            "url": url,
            "author": author,
            "author_url": author_url,
            "inserted_at": inserted_at,
            "published_at": article.get("published_at") or published_at,
            "media": media,
            "categories": categories,
            "attachments": attachments,
            "extra": extra,
            "language": language,
            "summary": summary,
            "cleaned_content": cleaned_content,
            "chunk_count": len(chunk_records),
            "chunks": chunk_records,
            "metadata": metadata,
            "processed_at": datetime.utcnow().isoformat() + "Z",
        }
        results_store[article_id] = result
        return result


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("deepseek-api-key")],
)
@modal.fastapi_endpoint(method="POST")
async def folo_webhook(request: Request) -> dict[str, Any]:
    """
    供 folo.is webhook 调用的入口。
    请求体示例：
    {
      "id": "post-123",
      "title": "...",
      "url": "https://...",
      "content": "文章全文",
      "published_at": "2026-04-20T12:00:00Z"
    }
    """
    content_type = request.headers.get("content-type", "").lower()
    raw_body = (await request.body()).decode("utf-8", errors="ignore").strip()
    if not raw_body:
        raise HTTPException(status_code=400, detail="Empty request body.")
    raw_payload: dict[str, Any]
    raw_payload = parse_webhook_payload(content_type, raw_body)
    payload = coerce_payload(raw_payload)
    content_value = payload.get("content")
    raw_summary = payload.get("summary")
    normalized_content = _normalize_plain_text(content_value)
    normalized_raw_summary = _normalize_plain_text(raw_summary)
    has_content = normalized_content is not None
    has_raw_summary = normalized_raw_summary is not None

    # 只有 raw_summary、没有 content 时，跳过后续分析和 research_reports 入库
    if (not has_content) and has_raw_summary:
        await save_report_to_neon.remote.aio(
            payload,
            {},
            raw_payload,
            "skipped",
            "Missing content; raw_summary-only payload skipped.",
        )
        return {
            "ok": True,
            "skipped": True,
            "reason": "raw_summary_only_no_content",
            "article_id": payload.get("id") or payload.get("url"),
            "model_version": ANALYSIS_MODEL_VERSION,
        }

    precheck = await check_report_processing_status.remote.aio(payload)
    if precheck.get("already_processed"):
        await save_report_to_neon.remote.aio(
            payload,
            {},
            raw_payload,
            "skipped",
            f"Already processed with same model_version: {ANALYSIS_MODEL_VERSION}",
        )
        return {
            "ok": True,
            "skipped": True,
            "reason": "already_processed_same_model_version",
            "article_id": payload.get("id") or payload.get("url"),
            "model_version": ANALYSIS_MODEL_VERSION,
            "neon": precheck,
        }

    try:
        result = await ArticlePipeline().run.remote.aio(payload)
        neon_result = await save_report_to_neon.remote.aio(payload, result, raw_payload, "success", "")
    except Exception as exc:
        await save_report_to_neon.remote.aio(payload, {}, raw_payload, "error", str(exc)[:2000])
        raise
    # 只返回摘要信息，避免 webhook 响应过大
    return {
        "ok": True,
        "article_id": result["article_id"],
        "chunk_count": result["chunk_count"],
        "neon": neon_result,
        # "tags": result["metadata"].get("tags", []),
        # "categories": result["metadata"].get("categories", []),
    }


@app.function(image=image)
def get_processed_article(article_id: str) -> dict[str, Any]:
    if article_id not in results_store:
        return {"ok": False, "error": "not_found", "article_id": article_id}
    return {"ok": True, "data": results_store[article_id]}


@app.function(image=image, secrets=[modal.Secret.from_name("neon-db-secret")])
def check_report_processing_status(article_data: dict[str, Any]) -> dict[str, Any]:
    import psycopg2
    from psycopg2.extras import RealDictCursor

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL")

    content_hash = build_content_hash(article_data)
    conn = psycopg2.connect(database_url)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    try:
        cur.execute(
            """
            SELECT rr.id AS report_id, ra.id AS analysis_id
            FROM research_reports rr
            JOIN report_analyses ra ON ra.report_id = rr.id
            WHERE rr.content_hash = %s
              AND ra.model_version = %s
              AND ra.is_active = TRUE
            ORDER BY ra.created_at DESC
            LIMIT 1
            """,
            (content_hash, ANALYSIS_MODEL_VERSION),
        )
        row = cur.fetchone()
        if row:
            return {
                "ok": True,
                "already_processed": True,
                "report_id": str(row["report_id"]),
                "analysis_id": str(row["analysis_id"]),
                "content_hash": content_hash,
                "model_version": ANALYSIS_MODEL_VERSION,
            }
        return {
            "ok": True,
            "already_processed": False,
            "content_hash": content_hash,
            "model_version": ANALYSIS_MODEL_VERSION,
        }
    finally:
        cur.close()
        conn.close()


@app.function(image=image, secrets=[modal.Secret.from_name("neon-db-secret")])
def save_report_to_neon(
    article_data: dict[str, Any],
    analysis_result: dict[str, Any],
    raw_payload: dict[str, Any] | None = None,
    status: str = "success",
    error_message: str = "",
) -> dict[str, Any]:
    import psycopg2
    from psycopg2.extras import Json, RealDictCursor

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Missing DATABASE_URL")

    cleaned_content = str(analysis_result.get("cleaned_content") or article_data.get("content") or "")
    content_hash = build_content_hash(article_data, cleaned_content=cleaned_content)
    metadata = analysis_result.get("metadata") if isinstance(analysis_result, dict) else {}
    metadata = normalize_metadata_output(metadata)
    chunks = analysis_result.get("chunks") if isinstance(analysis_result, dict) else []
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(chunks, list):
        chunks = []

    metadata_hash = stable_hash(metadata)
    chunk_signature = [
        {"chunk_index": c.get("chunk_index"), "text": c.get("text", ""), "embedding_dim": c.get("embedding_dim")}
        for c in chunks
        if isinstance(c, dict)
    ]
    chunk_hash = stable_hash(chunk_signature)

    conn = psycopg2.connect(database_url)
    cur = conn.cursor(cursor_factory=RealDictCursor)

    try:
        # 0) 原始审计日志：始终落库，便于排障
        cur.execute(
            """
            INSERT INTO raw_ingestion_log (article_id, payload, status, error_message)
            VALUES (%s, %s, %s, %s)
            """,
            (
                str(article_data.get("id") or article_data.get("article_id") or ""),
                Json(raw_payload or article_data),
                status,
                error_message[:2000] if error_message else None,
            ),
        )

        if status != "success":
            conn.commit()
            return {"ok": False, "status": status}

        # 1) 文章去重 Upsert（content_hash 唯一）
        cur.execute(
            """
            INSERT INTO research_reports (
                content_hash, article_id, title, url, author, author_url, language, source_org,
                source_title, published_at, inserted_at, media, categories, attachments, extra, raw_summary, cleaned_content
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_hash) DO UPDATE SET
                title = EXCLUDED.title,
                url = EXCLUDED.url,
                author = EXCLUDED.author,
                author_url = EXCLUDED.author_url,
                language = EXCLUDED.language,
                source_title = EXCLUDED.source_title,
                published_at = EXCLUDED.published_at,
                inserted_at = EXCLUDED.inserted_at,
                media = EXCLUDED.media,
                categories = EXCLUDED.categories,
                attachments = EXCLUDED.attachments,
                extra = EXCLUDED.extra,
                raw_summary = EXCLUDED.raw_summary,
                cleaned_content = EXCLUDED.cleaned_content
            RETURNING id
            """,
            (
                content_hash,
                analysis_result.get("article_id"),
                _normalize_plain_text(analysis_result.get("title")),
                _normalize_plain_text(analysis_result.get("url")),
                _normalize_plain_text(analysis_result.get("author")),
                _normalize_plain_text(analysis_result.get("author_url")),
                _normalize_plain_text(analysis_result.get("language")),
                _normalize_plain_text(analysis_result.get("source")) or "folo.is",
                _normalize_plain_text(analysis_result.get("source_title")),
                analysis_result.get("published_at"),
                analysis_result.get("inserted_at"),
                Json(_normalize_json_container(analysis_result.get("media"))),
                Json(_normalize_json_container(analysis_result.get("categories"))),
                Json(_normalize_json_container(analysis_result.get("attachments"))),
                Json(_normalize_json_container(analysis_result.get("extra"))),
                _normalize_plain_text(analysis_result.get("summary")),
                cleaned_content,
            ),
        )
        report_id = cur.fetchone()["id"]

        # 2) 如果当前激活版本且内容签名一致，直接跳过重复分析写入
        cur.execute(
            """
            SELECT id, metadata
            FROM report_analyses
            WHERE report_id = %s AND model_version = %s AND is_active = TRUE
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (report_id, ANALYSIS_MODEL_VERSION),
        )
        existing = cur.fetchone()
        if existing:
            existing_meta = existing.get("metadata") or {}
            existing_meta_hash = str(existing_meta.get("_metadata_hash", ""))
            existing_chunk_hash = str(existing_meta.get("_chunk_hash", ""))
            if existing_meta_hash == metadata_hash and existing_chunk_hash == chunk_hash:
                conn.commit()
                return {
                    "ok": True,
                    "report_id": str(report_id),
                    "analysis_id": str(existing["id"]),
                    "deduplicated": True,
                    "reason": "same_active_version_and_same_content_signature",
                }

        # 3) 旧版本置为非激活，并写入新分析版本
        cur.execute(
            """
            UPDATE report_analyses
            SET is_active = FALSE
            WHERE report_id = %s AND model_version = %s AND is_active = TRUE
            """,
            (report_id, ANALYSIS_MODEL_VERSION),
        )

        metadata_with_version = dict(metadata)
        metadata_with_version["_metadata_hash"] = metadata_hash
        metadata_with_version["_chunk_hash"] = chunk_hash
        metadata_with_version["_embedding_model"] = EMBEDDING_MODEL_NAME
        metadata_with_version["_embedding_dim"] = EMBEDDING_DIMENSION
        metadata_with_version["_metadata_model"] = METADATA_MODEL_NAME
        metadata_with_version["_schema_version"] = METADATA_SCHEMA_VERSION
        metadata_with_version["_analysis_model_version"] = ANALYSIS_MODEL_VERSION

        cur.execute(
            """
            INSERT INTO report_analyses (report_id, model_version, llm_summary, metadata, sentiment_score, is_active)
            VALUES (%s, %s, %s, %s, %s, TRUE)
            RETURNING id
            """,
            (
                report_id,
                ANALYSIS_MODEL_VERSION,
                _normalize_metadata_text(metadata.get("summary") or analysis_result.get("summary")),
                Json(metadata_with_version),
                None,
            ),
        )
        analysis_id = cur.fetchone()["id"]

        # 4) 写入 chunk 向量（绑定到 analysis 版本）
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            embedding = chunk.get("embedding")
            if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMENSION:
                continue
            vector_literal = "[" + ",".join(str(float(v)) for v in embedding) + "]"
            cur.execute(
                """
                INSERT INTO report_chunks (analysis_id, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s::vector)
                """,
                (
                    analysis_id,
                    int(chunk.get("chunk_index") or 0),
                    str(chunk.get("text") or ""),
                    vector_literal,
                ),
            )

        conn.commit()

        # 每天最多清理一次 raw_ingestion_log（best-effort，不影响主流程）
        try:
            today_key = datetime.utcnow().date().isoformat()
            if cleanup_store.get("raw_ingestion_log_last_cleanup_date") != today_key:
                cur.execute("DELETE FROM raw_ingestion_log WHERE created_at < NOW() - INTERVAL '1 day'")
                conn.commit()
                cleanup_store["raw_ingestion_log_last_cleanup_date"] = today_key
        except Exception:
            pass
        return {
            "ok": True,
            "report_id": str(report_id),
            "analysis_id": str(analysis_id),
            "deduplicated": False,
            "model_version": ANALYSIS_MODEL_VERSION,
            "chunk_count": len(chunks),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()