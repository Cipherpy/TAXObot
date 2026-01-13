import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from docx import Document

# Matches:
# Species Name: ...
# species   name : ...
# SPECIES NAME: ...
SPECIES_PATTERN = re.compile(r"^\s*species\s*name\s*:\s*(.+)$", re.IGNORECASE)


def extract_genus(species_name: str) -> Optional[str]:
    """
    Extract genus as the first token of the species name string.
    Example: 'Harmothoe saldanha Day, 1953' -> 'Harmothoe'
    """
    if not species_name:
        return None
    # Take first word token
    genus = species_name.strip().split()[0]
    # Basic sanity check: Genus usually starts with uppercase
    return genus if genus and genus[0].isalpha() else None


def chunk_docx_by_species_name(docx_path: str) -> List[Dict]:
    """
    Chunk a .docx by 'Species Name:' markers (case-insensitive) and attach metadata.
    """
    doc = Document(docx_path)
    paras = [p.text.strip() for p in doc.paragraphs]

    chunks: List[Dict] = []
    current_lines = []
    current_species = None
    start_idx = None

    def flush(end_idx: int):
        nonlocal current_lines, current_species, start_idx
        if current_species and current_lines:
            genus = extract_genus(current_species)

            chunks.append({
                "species_name": current_species,
                "genus": genus,
                "authority": "CMLRE",
                "year": 2025,
                "section": "taxonomic information",
                "text": "\n".join(current_lines).strip(),
                "source_file": Path(docx_path).name,
                "paragraph_span": [start_idx, end_idx],
            })

        current_lines = []
        current_species = None
        start_idx = None

    for i, text in enumerate(paras):
        if not text:
            continue

        # Case-insensitive check: normalize to lowercase for matching
        if SPECIES_PATTERN.match(text.lower()):
            # close previous chunk
            flush(end_idx=i - 1)

            # start new chunk
            current_lines = [text]  # keep original
            start_idx = i

            # extract the species name from the original text
            m = SPECIES_PATTERN.match(text)
            current_species = m.group(1).strip() if m else "UNKNOWN"
        else:
            if current_species is not None:
                current_lines.append(text)

    flush(end_idx=len(paras) - 1)
    return chunks


def save_jsonl(chunks: List[Dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    docx_file = "/home/reshma/TAXObot/Chunking/docs/Polychaetes_data.docx"
    out_file = "species_chunks_with_metadata.jsonl"

    chunks = chunk_docx_by_species_name(docx_file)
    print("Total chunks:", len(chunks))

    if chunks:
        print("\nExample chunk metadata:")
        for k in ["species_name", "genus", "authority", "year", "section"]:
            print(f"{k}: {chunks[0].get(k)}")

    save_jsonl(chunks, out_file)
    print("\nSaved:", out_file)
