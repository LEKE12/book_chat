# loader.py
from __future__ import annotations

import re
import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict, Tuple

from docling.document_converter import DocumentConverter


# ============================================================
# Config
# ============================================================

@dataclass
class ChunkConfig:
    max_tokens: int = 800
    min_tokens: int = 200
    overlap_tokens: int = 120
    keep_heading: bool = True


# ============================================================
# Token approximation
# ============================================================

def approx_tokens(text: str) -> int:
    """Rough estimate: ~4 chars/token for English-ish text."""
    text = text or ""
    return max(1, len(text) // 4)


# ============================================================
# Markdown splitting
# ============================================================

_HEADING_RE = re.compile(r"^(#{1,6}\s+.*)$", re.MULTILINE)

def normalize_md(md: str) -> str:
    """Light normalization to reduce weird whitespace artifacts."""
    md = (md or "").replace("\r\n", "\n")
    md = md.replace("\t", " ")
    md = re.sub(r"[ ]{2,}", " ", md)  # collapse repeated spaces
    return md.strip()

def split_markdown_by_headings(md: str) -> List[Tuple[Optional[str], str]]:
    """
    Return blocks as (heading_line, content_until_next_heading).
    heading_line includes the full markdown heading, e.g. "## Chapter 1".
    If no headings exist, returns [(None, whole_text)].
    """
    md = normalize_md(md)

    matches = list(_HEADING_RE.finditer(md))
    if not matches:
        return [(None, md.strip())]

    blocks: List[Tuple[Optional[str], str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        heading = m.group(1).strip()
        content = md[start:end].strip()
        blocks.append((heading, content))
    return blocks

def split_into_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n+", text or "") if p.strip()]

def split_into_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter.
    If you need better sentence boundaries later, swap this for spaCy.
    """
    text = re.sub(r"\s+", " ", (text or "")).strip()
    if not text:
        return []
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])", text)
    return [s.strip() for s in sents if s.strip()]

def apply_overlap(chunks: List[str], overlap_tokens: int) -> List[str]:
    if overlap_tokens <= 0 or len(chunks) <= 1:
        return chunks

    out = [chunks[0]]
    overlap_chars = overlap_tokens * 4

    for i in range(1, len(chunks)):
        prev = out[-1]
        curr = chunks[i]
        tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
        out.append((tail + "\n" + curr).strip())

    return out


# ============================================================
# Recursive chunking (heading -> paragraphs -> sentences)
# ============================================================

def recursive_chunk_section(
    heading: Optional[str],
    text: str,
    cfg: ChunkConfig,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    prefix = (heading + "\n") if (cfg.keep_heading and heading) else ""

    # Whole section fits
    if approx_tokens(prefix + text) <= cfg.max_tokens:
        return [(prefix + text).strip()]

    paras = split_into_paragraphs(text)
    chunks: List[str] = []
    buf: List[str] = []

    def flush_buf():
        if not buf:
            return
        chunks.append((prefix + "\n\n".join(buf)).strip())
        buf.clear()

    for p in paras:
        # Paragraph too big => sentences
        if approx_tokens(p) > cfg.max_tokens:
            flush_buf()

            sentences = split_into_sentences(p)
            sent_buf: List[str] = []

            for s in sentences:
                candidate = " ".join(sent_buf + [s])
                if approx_tokens(prefix + candidate) <= cfg.max_tokens:
                    sent_buf.append(s)
                else:
                    if sent_buf:
                        chunks.append((prefix + " ".join(sent_buf)).strip())
                        sent_buf = [s]
                    else:
                        # Single sentence too large: hard cut by chars
                        max_chars = cfg.max_tokens * 4
                        for i in range(0, len(s), max_chars):
                            chunks.append((prefix + s[i:i + max_chars]).strip())
                        sent_buf = []

            if sent_buf:
                chunks.append((prefix + " ".join(sent_buf)).strip())

        else:
            candidate = "\n\n".join(buf + [p])
            if approx_tokens(prefix + candidate) <= cfg.max_tokens:
                buf.append(p)
            else:
                flush_buf()
                buf.append(p)

                # Safety
                if approx_tokens(prefix + "\n\n".join(buf)) > cfg.max_tokens:
                    flush_buf()

    flush_buf()

    # Merge tiny chunks if possible
    merged: List[str] = []
    for ch in chunks:
        if not merged:
            merged.append(ch)
            continue

        if approx_tokens(merged[-1]) < cfg.min_tokens:
            candidate = merged[-1] + "\n\n" + ch
            if approx_tokens(candidate) <= cfg.max_tokens:
                merged[-1] = candidate
            else:
                merged.append(ch)
        else:
            merged.append(ch)

    return apply_overlap(merged, cfg.overlap_tokens)

def recursive_chunk_markdown(md: str, cfg: ChunkConfig) -> List[Dict]:
    blocks = split_markdown_by_headings(md)
    out: List[Dict] = []
    chunk_id = 0

    for heading, content in blocks:
        for ch in recursive_chunk_section(heading, content, cfg):
            out.append({
                "chunk_id": chunk_id,
                "heading": heading,
                "text": ch,
                "approx_tokens": approx_tokens(ch),
            })
            chunk_id += 1

    return out


# ============================================================
# File helpers
# ============================================================

def stable_id(*parts: str) -> str:
    """Deterministic hash helper (useful for vector DB upserts)."""
    h = hashlib.sha1()
    joined = "||".join([p for p in parts if p is not None])
    h.update(joined.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def save_jsonl(rows: List[Dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sanitize_stem(stem: str) -> str:
    """
    Makes filenames safe across OSes:
    - removes problematic chars
    - collapses whitespace
    """
    stem = stem.strip()
    stem = re.sub(r"[\/\\:\*\?\"<>\|]+", "-", stem)  # Windows-illegal chars
    stem = re.sub(r"\s+", " ", stem)
    return stem


# ============================================================
# 1) PDF -> Markdown into /data (same name as PDF)
# ============================================================

def pdfs_to_markdown(books_dir: Path, data_dir: Path) -> int:
    books_dir = books_dir.expanduser().resolve()
    data_dir = data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    if not books_dir.exists():
        raise FileNotFoundError(f"Books dir not found: {books_dir}")

    pdf_files = sorted(books_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {books_dir}")
        return 0

    converter = DocumentConverter()
    count = 0

    for pdf_path in pdf_files:
        safe_stem = sanitize_stem(pdf_path.stem)
        out_md = data_dir / f"{safe_stem}.md"

        try:
            doc = converter.convert(str(pdf_path)).document
            md_text = doc.export_to_markdown()
            md_text = normalize_md(md_text)
            out_md.write_text(md_text, encoding="utf-8")
            print(f"✅ {pdf_path.name} -> {out_md.name}")
            count += 1
        except Exception as e:
            print(f"❌ Failed {pdf_path.name}: {e}")

    return count


# ============================================================
# 2) Chunk each .md in /data into per-book JSONL
# ============================================================

def chunk_markdown_files_in_dir(data_dir: Path, cfg: ChunkConfig) -> int:
    """
    For each .md in data_dir:
      write {data_dir}/{same_name}_chunks.jsonl
    """
    data_dir = data_dir.expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(data_dir.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {data_dir}")
        return 0

    total_chunks = 0

    for md_path in md_files:
        md_text = read_text_file(md_path)
        chunks = recursive_chunk_markdown(md_text, cfg)

        doc_id = stable_id(str(md_path))
        title = md_path.stem

        enriched: List[Dict] = []
        for c in chunks:
            chunk_uid = stable_id(doc_id, str(c["chunk_id"]), c.get("heading") or "")
            enriched.append({
                "doc_id": doc_id,
                "source_path": str(md_path),
                "title": title,
                **c,
                "chunk_uid": chunk_uid,
                "chunk_config": asdict(cfg),
            })

        out_jsonl = data_dir / f"{md_path.stem}_chunks.jsonl"
        save_jsonl(enriched, out_jsonl)

        total_chunks += len(enriched)
        print(f"📦 {md_path.name} -> {out_jsonl.name} ({len(enriched)} chunks)")

    return total_chunks


# ============================================================
# Main:
# ============================================================

def main():
    # ✅ Hard-coded base directory (your request)
    base_dir = Path("/Users/lekeadako/Documents/portfolio projects/Book_chat")

    books_dir = base_dir / "books"
    data_dir = base_dir / "data"

    cfg = ChunkConfig(
        max_tokens=800,
        min_tokens=200,
        overlap_tokens=120,
        keep_heading=True,
    )

    print("\n📘 Step 1/2: Converting PDFs in /books → Markdown in /data ...")
    n_pdfs = pdfs_to_markdown(books_dir, data_dir)
    print(f"✔ Converted {n_pdfs} PDF(s).")

    print("\n✂️ Step 2/2: Chunking Markdown files in /data ...")
    total_chunks = chunk_markdown_files_in_dir(data_dir, cfg)
    print(f"✔ Wrote {total_chunks} total chunks.")

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
