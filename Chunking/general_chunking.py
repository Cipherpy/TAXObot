import json
from pathlib import Path
from typing import List, Dict, Optional
from docx import Document


def is_bold_paragraph(p, bold_ratio_threshold: float = 0.7) -> bool:
    """
    Returns True if a paragraph is "mostly bold".
    Works even if only part of the paragraph is bold (common in Word).
    """
    text = (p.text or "").strip()
    if not text:
        return False

    # If it is a Heading style, treat as heading/boundary too
    style_name = (p.style.name or "").lower() if p.style else ""
    if "heading" in style_name:
        return True

    runs = [r for r in p.runs if (r.text or "").strip()]
    if not runs:
        return False

    total_chars = sum(len(r.text.strip()) for r in runs)
    bold_chars = sum(len(r.text.strip()) for r in runs if r.bold)

    if total_chars == 0:
        return False

    return (bold_chars / total_chars) >= bold_ratio_threshold


def chunk_docx_by_bold_headings(
    docx_path: str,
    bold_ratio_threshold: float = 0.7,
    include_bold_heading_in_text: bool = True,
) -> List[Dict]:
    """
    Chunk a .docx by bold paragraphs:
    - A new chunk starts at a bold paragraph.
    - Chunk continues until before the next bold paragraph.
    - Metadata section is set to 'general information'.
    """
    doc = Document(docx_path)
    paras = list(doc.paragraphs)

    chunks: List[Dict] = []
    current_lines: List[str] = []
    current_title: Optional[str] = None
    start_idx: Optional[int] = None

    def flush(end_idx: int):
        nonlocal current_lines, current_title, start_idx
        if current_title and current_lines and start_idx is not None:
            chunks.append({
                "title": current_title,                 # the bold sentence
                "section": "general information",
                "text": "\n".join(current_lines).strip(),
                "source_file": Path(docx_path).name,
                "paragraph_span": [start_idx, end_idx],
            })
        current_lines = []
        current_title = None
        start_idx = None

    for i, p in enumerate(paras):
        text = (p.text or "").strip()
        if not text:
            continue

        if is_bold_paragraph(p, bold_ratio_threshold=bold_ratio_threshold):
            # New chunk boundary -> flush previous
            flush(end_idx=i - 1)

            # Start a new chunk
            current_title = text
            start_idx = i
            current_lines = [text] if include_bold_heading_in_text else []
        else:
            # Add content only if we already started a chunk
            if current_title is not None:
                current_lines.append(text)
            # If text appears before the first bold heading, ignore by design

    # Flush last chunk
    flush(end_idx=len(paras) - 1)

    return chunks


def save_jsonl(chunks: List[Dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    docx_file = "/home/reshma/TAXObot/Chunking/docs/General Information.docx"
    out_file = "general_info_chunks.jsonl"

    chunks = chunk_docx_by_bold_headings(
        docx_path=docx_file,
        bold_ratio_threshold=0.7,          # lower to 0.6 if headings are only partially bold
        include_bold_heading_in_text=True  # keep bold sentence inside the chunk text
    )

    print("Total chunks:", len(chunks))
    if chunks:
        print("\nExample metadata:")
        print("title:", chunks[0]["title"])
        print("section:", chunks[0]["section"])
        print("preview:\n", chunks[0]["text"][:400], "...")

    save_jsonl(chunks, out_file)
    print("\nSaved:", out_file)
