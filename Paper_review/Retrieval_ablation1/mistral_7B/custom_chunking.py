import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from docx import Document


# -----------------------------
# 1) Species marker (robust)
# -----------------------------
SPECIES_PATTERN = re.compile(
    r"^\s*species\s*name\s*[:\-]\s*(.+)$",
    re.IGNORECASE
)


def is_species_marker(text: str) -> bool:
    return bool(SPECIES_PATTERN.match((text or "").strip()))

def extract_species_name(text: str) -> str:
    m = SPECIES_PATTERN.match((text or "").strip())
    return m.group(1).strip() if m else "UNKNOWN"

def extract_genus(species_name: str) -> Optional[str]:
    if not species_name:
        return None
    parts = species_name.strip().split()
    return parts[0] if parts else None


# -----------------------------
# 2) Robust heading detection
#    (bold OR heading style OR "Label:" pattern)
# -----------------------------
LABEL_HEADING_PATTERN = re.compile(
    r"^[A-Za-z][A-Za-z0-9\s\-/]{1,60}:\s*$"  # e.g. "Body:" "Prostomium:" "Antennae and cirri:"
)

def run_is_bold(run) -> bool:
    # run.bold can be True/False/None
    # run.font.bold sometimes carries inherited bold
    return bool(run.bold) or bool(getattr(run.font, "bold", False))

def is_heading_style(p) -> bool:
    style_name = (p.style.name or "").lower() if p.style else ""
    return "heading" in style_name

def is_boldish_paragraph(p, bold_ratio_threshold: float = 0.35) -> bool:
    """
    Bold detector that works better on real docx:
    - considers run.bold OR run.font.bold
    - uses a lower default threshold
    """
    text = (p.text or "").strip()
    if not text:
        return False

    runs = [r for r in p.runs if (r.text or "").strip()]
    if not runs:
        return False

    total_chars = sum(len(r.text.strip()) for r in runs)
    if total_chars == 0:
        return False

    bold_chars = sum(len(r.text.strip()) for r in runs if run_is_bold(r))
    return (bold_chars / total_chars) >= bold_ratio_threshold

def looks_like_label_heading(text: str) -> bool:
    """
    Catch headings that are not bold in docx after merge:
    'Body:' 'Prostomium:' etc.
    """
    t = (text or "").strip()
    if not t:
        return False
    # keep it short (avoid treating long sentences as headings)
    if len(t) > 80:
        return False
    return bool(LABEL_HEADING_PATTERN.match(t))

def is_general_heading_paragraph(p, bold_ratio_threshold: float = 0.35) -> bool:
    """
    A paragraph is a general-heading boundary if:
    - Heading style, OR
    - Boldish, OR
    - Looks like "Label:" (fallback)
    But not if it is "Species Name:"
    """
    text = (p.text or "").strip()
    if not text:
        return False
    if is_species_marker(text):
        return False

    return (
        is_heading_style(p)
        or is_boldish_paragraph(p, bold_ratio_threshold=bold_ratio_threshold)
        or looks_like_label_heading(text)
    )


# -----------------------------
# 3) Combined chunker
# -----------------------------
def chunk_merged_docx(
    docx_path: str,
    authority: str = "CMLRE",
    year: int = 2025,
    bold_ratio_threshold: float = 0.35,
    include_heading_in_general_text: bool = True,
) -> List[Dict]:
    """
    For ONE merged docx:
    - Taxonomic chunk per species block (Species Name -> next Species Name)
    - General info chunks within each species block (heading/bold/label -> next heading/bold/label)
    - Also captures general-info chunks that appear before the first species (species_name=None)
    """
    doc = Document(docx_path)
    paras = list(doc.paragraphs)
    source_file = Path(docx_path).name

    # Find all species marker indices
    species_starts = [i for i, p in enumerate(paras) if is_species_marker((p.text or "").strip())]

    chunks: List[Dict] = []

    # ---------- Helper: build general chunks within a range ----------
    def add_general_chunks_in_range(start_i: int, end_i: int, species_name: Optional[str], genus: Optional[str]):
        # heading boundaries within [start_i, end_i]
        headings = []
        for i in range(start_i, end_i + 1):
            txt = (paras[i].text or "").strip()
            if not txt:
                continue
            if is_general_heading_paragraph(paras[i], bold_ratio_threshold=bold_ratio_threshold):
                headings.append(i)

        for k, h_start in enumerate(headings):
            h_end = (headings[k + 1] - 1) if (k + 1 < len(headings)) else end_i
            title = (paras[h_start].text or "").strip()

            lines = []
            for j in range(h_start, h_end + 1):
                t = (paras[j].text or "").strip()
                if not t:
                    continue
                if j == h_start:
                    if include_heading_in_general_text:
                        lines.append(t)
                else:
                    lines.append(t)

            text_block = "\n".join(lines).strip()
            if text_block:
                chunks.append({
                    "chunk_type": "general",
                    "species_name": species_name,
                    "genus": genus,
                    "authority": authority,
                    "year": year,
                    "section": "general information",
                    "title": title,
                    "text": text_block,
                    "source_file": source_file,
                    "paragraph_span": [h_start, h_end],
                })

    # ---------- (A) General info BEFORE first species ----------
    if species_starts:
        pre_end = species_starts[0] - 1
        if pre_end >= 0:
            add_general_chunks_in_range(0, pre_end, species_name=None, genus=None)

    # ---------- (B) Species blocks ----------
    for idx, s_start in enumerate(species_starts):
        s_end = (species_starts[idx + 1] - 1) if (idx + 1 < len(species_starts)) else (len(paras) - 1)

        marker_text = (paras[s_start].text or "").strip()
        species_name = extract_species_name(marker_text)
        genus = extract_genus(species_name)

        # Taxonomic chunk: whole species block
        block_lines = []
        for j in range(s_start, s_end + 1):
            t = (paras[j].text or "").strip()
            if t:
                block_lines.append(t)

        tax_text = "\n".join(block_lines).strip()
        chunks.append({
            "chunk_type": "taxonomic",
            "species_name": species_name,
            "genus": genus,
            "authority": authority,
            "year": year,
            "section": "taxonomic information",
            "text": tax_text,
            "source_file": source_file,
            "paragraph_span": [s_start, s_end],
        })

        # General chunks inside this species block (usually after Species Name line)
        if s_start + 1 <= s_end:
            add_general_chunks_in_range(s_start + 1, s_end, species_name=species_name, genus=genus)

    return chunks


def save_jsonl(chunks: List[Dict], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    docx_file = "/home/reshma/TAXObot/Chunking/docs/TAXObot Corpus.docx"  # <-- your merged file
    out_file = "merged_taxonomic_general_chunks.jsonl"

    chunks = chunk_merged_docx(
        docx_path=docx_file,
        authority="CMLRE",
        year=2025,
        bold_ratio_threshold=0.35,  # try 0.25 if still missing headings
        include_heading_in_general_text=True,
    )

    tax = sum(1 for c in chunks if c["chunk_type"] == "taxonomic")
    gen = sum(1 for c in chunks if c["chunk_type"] == "general")

    print("Total chunks:", len(chunks))
    print("Taxonomic chunks:", tax)
    print("General chunks:", gen)

    save_jsonl(chunks, out_file)
    print("Saved:", out_file)
