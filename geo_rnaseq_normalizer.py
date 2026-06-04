from __future__ import annotations
from pathlib import Path
from urllib.parse import urljoin, parse_qs, urlparse
import requests
from bs4 import BeautifulSoup
import concurrent.futures
import getpass
import gzip
import io
import json
import os
import re
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import time
import warnings
import numpy as np
import GEOparse
import pandas as pd
import requests as _requests
from groq import Groq

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
TABULAR_EXTS = {".txt", ".csv", ".tsv", ".tab"}
_CHUNK   = 64 * 1024 * 1024   # 64 MB read buffer for faster downloads
_WORKERS = 64

_groq_client: Optional[Groq] = None


# ────────────
# Groq client
# ────────────

def _get_client() -> Groq:
    global _groq_client
    if _groq_client is not None:
        return _groq_client
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("Groq API key not found in environment.")
        print("Get a free key at: https://console.groq.com → API Keys")
        api_key = getpass.getpass("Paste your Groq API key: ").strip()
    if not api_key:
        raise ValueError("No Groq API key provided.")
    os.environ["GROQ_API_KEY"] = api_key
    _groq_client = Groq(api_key=api_key)
    print("✓ Groq client ready.\n")
    return _groq_client

def to_cpm(df: pd.DataFrame) -> pd.DataFrame:
    """Divide each column by its sum and scale to 1e6."""
    # Ensure only numeric columns
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    col_sums = numeric_df.sum(axis=0).replace(0, np.nan)
    return numeric_df.divide(col_sums, axis=1) * 1e6

def to_log2cpm(cpm: pd.DataFrame) -> pd.DataFrame:
    """log2(CPM + 1)."""
    return np.log2(cpm + 1)

def _detect_log_base(df: pd.DataFrame) -> str:
    """Heuristic: max < 35 → log2, else natural log."""
    return "log2" if df.max().max() < 35 else "loge"

def convert_to_cpm(df: pd.DataFrame, norm_type: str) -> tuple[pd.DataFrame, str]:
    nt = norm_type.lower()

    if nt == "raw_counts":
        print("  [convert] raw_counts → CPM")
        return to_cpm(df), "raw_counts"

    elif nt == "rpkm_fpkm":
        print("  [convert] FPKM/RPKM → CPM")
        return df, "rpkm_fpkm"

    elif nt == "tpm":
        print("  [convert] TPM → CPM (re-scaling column sums to 1e6)")
        return df, "tpm"

    elif nt == "cpm":
        print("  [convert] CPM → CPM (re-scaling to correct any drift)")
        return to_cpm(df), "cpm"

    elif nt == "log_transformed":
        base = _detect_log_base(df)
        if base == "log2":
            print("  [convert] log_transformed → reversing log2 (2^x − 1) → CPM")
            linear = (2 ** df) - 1
        else:
            print("  [convert] log_transformed → reversing ln (e^x − 1) → CPM")
            linear = np.expm1(df)
        return to_cpm(linear.clip(lower=0)), "log_transformed"

    elif nt in ("vst_rlog", "tmm_rle"):
        print(f"  [convert] {norm_type} → treating as pseudo-counts → CPM")
        return to_cpm(df), nt

    else:
        warnings.warn(
            f"\n Cannot convert '{norm_type}' to CPM — values returned as-is.\n"
            f"   log2(x+1) will still be applied but the result is not CPM.",
            stacklevel=3,
        )
        return df, f"{norm_type}_unconverted"

@dataclass
class GseResult:
    """Everything produced for one GEO accession."""
    accession: str
    normalization_type: str   
    effective_norm: str       
    confidence: str
    reasoning: str

    counts_df:   Optional[pd.DataFrame] = field(default=None, repr=False)
    cpm_df:      Optional[pd.DataFrame] = field(default=None, repr=False)
    log2cpm_df:  Optional[pd.DataFrame] = field(default=None, repr=False)

    @property
    def fpkm_conversion_failed(self) -> bool:
        """True if FPKM->CPM fell back due to missing gene lengths."""
        return "fallback" in self.effective_norm

    def save(self, output_dir: str | Path = ".") -> list[Path]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []

        target_df   = self.log2cpm_df if self.log2cpm_df is not None else self.counts_df
        target_name = "log2CPM" if self.log2cpm_df is not None else "counts"

        if target_df is not None:
            p = output_dir / f"{self.accession}_{target_name}.txt"
            target_df.to_csv(p, sep="\t")
            print(f"✓ {target_name} matrix → {p}  "
                  f"({target_df.shape[0]} genes × {target_df.shape[1]} samples)")
            written.append(p)

        mp = output_dir / f"{self.accession}_meta.json"
        with open(mp, "w") as f:
            json.dump({
                "accession":          self.accession,
                "normalization_type": self.normalization_type,
                "effective_norm":     self.effective_norm,
                "confidence":         self.confidence,
                "reasoning":          self.reasoning,
            }, f, indent=2)
        written.append(mp)
        return written

    def __repr__(self) -> str:
        df    = self.log2cpm_df or self.counts_df
        shape = f"{df.shape[0]}g × {df.shape[1]}s" if df is not None else "no matrix"
        return (f"GseResult(acc={self.accession!r}, "
                f"orig_norm={self.normalization_type!r}, "
                f"effective={self.effective_norm!r}, "
                f"conf={self.confidence!r}, matrix={shape})")



_NORM_SYSTEM = """You are a bioinformatics expert specializing in GEO datasets.
Classify the normalization type from a GSM data_processing section.
Return ONLY valid JSON, no markdown:
{
  "normalization_type": "<type>",
  "confidence": "<high|medium|low>",
  "reasoning": "<brief>"
}
Types: raw_counts | cpm | rpkm_fpkm | tpm | log_transformed | vst_rlog |
       quantile | tmm_rle | microarray_rma | fold_change | z_score | unknown
Focus on what is stored in the final supplementary files."""


def classify_normalization(dp_text: str) -> dict:
    resp = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _NORM_SYSTEM},
            {"role": "user",   "content": f"Classify:\n\n{dp_text}"},
        ],
        temperature=0, max_tokens=300,
    )
    text = resp.choices[0].message.content.strip()
    text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    return json.loads(text)


_COL_SYSTEM = """You are a bioinformatics expert.
A per-sample RNAseq file has these columns (after the gene-ID column):
{cols}
Return ONLY the name of the single column that contains the primary
expression values (raw counts, FPKM, TPM, or CPM — whichever is present).
Return just the column name, nothing else."""


def pick_count_column(columns: list[str], filename: str) -> str:
    prompt = _COL_SYSTEM.format(cols="\n".join(f"  - {c}" for c in columns))
    resp = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": f"File: {filename}\nWhich column has expression values?"},
        ],
        temperature=0, max_tokens=50,
    )
    chosen = resp.choices[0].message.content.strip().strip('"').strip("'")
    for c in columns:
        if c.lower() == chosen.lower():
            return c
    print(f"  [warn] Could not match '{chosen}' to columns; using '{columns[0]}'")
    return columns[0]


def _download_bytes(url: str) -> bytes:
    url = url.replace("ftp://", "https://", 1)
    resp = _requests.get(url, timeout=300)
    resp.raise_for_status()
    return resp.content


def _download_all(urls: list[str]) -> dict[str, bytes]:
    results: dict[str, bytes] = {}
    print(f"  Downloading {len(urls)} file(s) in parallel…")
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(_WORKERS, len(urls))) as ex:
        fut_to_url = {ex.submit(_download_bytes, u): u for u in urls}
        for fut in concurrent.futures.as_completed(fut_to_url):
            url = fut_to_url[fut]
            fname = url.split("/")[-1]
            try:
                results[url] = fut.result()
                print(f"    ✓ {fname} ({len(results[url]) / 1e6:.1f} MB)")
            except Exception as e:
                print(f"    ✗ {fname}: {e}")
    return results


def _is_tabular(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(e) or lower.endswith(e + ".gz") for e in TABULAR_EXTS)


def _decompress(raw: bytes, name: str) -> tuple[bytes, str]:
    if name.lower().endswith(".gz"):
        return gzip.decompress(raw), name[:-3]
    return raw, name


def _detect_sep(name: str) -> str:
    return "," if name.lower().endswith(".csv") else "\t"


def _parse_tabular(raw: bytes, filename: str) -> tuple[pd.Series, str]:
    sep = _detect_sep(filename)
    df  = pd.read_csv(
        io.StringIO(raw.decode("utf-8", errors="replace")),
        sep=sep, comment="#", header=0,
    )

    num_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().all()]
    str_cols = [c for c in df.columns if c not in num_cols]

    # Gene index from first non-numeric column
    gene_index = df[str_cols[0]].astype(str) if str_cols else df.index.astype(str)

    if len(num_cols) == 0:
        raise ValueError(f"No numeric columns in {filename}")
    elif len(num_cols) == 1:
        count_col = num_cols[0]
    else:
        count_col = pick_count_column(num_cols, filename)

    s = pd.to_numeric(df[count_col], errors="coerce")
    s.index = gene_index

    stem = re.sub(r"\.gz$", "", filename, flags=re.IGNORECASE)
    stem = re.sub(r"\.(csv|tsv|txt|tab)$", "", stem, flags=re.IGNORECASE)
    return s, stem

def _download_raw(
    accession: str,
    file_meta: list[dict],
    cache_dir: Path,
) -> int:

    page_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={accession}"

    print(f"  Scraping GEO download page: {accession}")

    r = requests.get(page_url, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    found = 0
    new_meta = []

    for a in soup.find_all("a", href=True):

        href = a["href"]

        parsed = urlparse(href)
        qs = parse_qs(parsed.query)

        filename = qs.get("file", [None])[0]

        if not filename:
            continue

        fname_lower = filename.lower()

        # require real count/raw files
        if not any(
            k in fname_lower
            for k in ["raw"]
        ):
            continue

        # exclude annotation files
        if any(
            bad in fname_lower
            for bad in ["annot", "annotation"]
        ):
            continue

        # only tabular-ish files
        if not fname_lower.endswith(
            (
                ".csv",
                ".csv.gz",
                ".tsv",
                ".tsv.gz",
                ".txt",
                ".txt.gz",
                ".tar",
                ".tar.gz",
            )
        ):
            continue
        
        full_url = urljoin(page_url, href)
        meta = {
            "url": full_url,
            "filename": filename,
            "is_tar": filename.endswith((".tar", ".tar.gz")),
            "cache_path": cache_dir / filename,
        }

        new_meta.append(meta)

        found += 1

        print(f"    ✓ found: {filename}")

    # replace old GEOparse-discovered files
    if new_meta:
        file_meta.clear()
        file_meta.extend(new_meta)

    print(f"  ✓ added {found} raw/count file(s)")

    return found

def _annotate_raw(
    accession: str,
    file_meta: list[dict],
    counts_df: pd.DataFrame,
) -> pd.DataFrame: # Changed to return the modified DataFrame

    print(f"  Fetching annotation for {accession}…")

        
    page_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={accession}"
    r = requests.get(page_url, timeout=60)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")

    # Gather all file links and names from the page
    available_files = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        # Filter for typical file downloads
        if "file=" in href:
            filename = href.split("file=")[-1]
            available_files.append({"filename": filename, "url": href})

    if not available_files:
        print("No files found on the GEO page.")
        return counts_df

    # 2. Use Groq to identify which filename is likely the annotation file
    prompt = f"""
    You are an expert bioinformatician. Look at this list of files available for download from NCBI GEO for accession {accession}:
    {json.dumps([f['filename'] for f in available_files])}

    Identify which file is most likely to be the annotation file (e.g., contains gene symbols, probe IDs, platform data, .soft, _family, or annotations).
    Return your answer strictly as a JSON object with the key "annotation_file". If none look like annotation files, return null.
    Example output: {{"annotation_file": "GSE1234_family.soft.gz"}}
    """

    chat_completion = _get_client().chat.completions.create(
        messages=[
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.1-8b-instant",
        response_format={"type": "json_object"}
    )
    
    response_data = json.loads(chat_completion.choices[0].message.content)
    target_filename = response_data.get("annotation_file")

    if not target_filename:
        print("Groq could not identify an annotation file from the filenames.")
        return counts_df

    print(f"--> Groq identified annotation file: {target_filename}")

    # Find the corresponding URL
    target_url = next((f["url"] for f in available_files if f["filename"] == target_filename), None)
    if not target_url:
        return counts_df

    # 3. Download and load the annotation file into a DataFrame
    print(f"Downloading {target_filename}...")
    # fix relative/bare URLs
    if target_url.startswith("http"):
        pass  # already absolute
    elif target_url.startswith("/"):
        target_url = "https://www.ncbi.nlm.nih.gov" + target_url
    else:
        # bare filename — reconstruct the GEO download URL
        target_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={accession}&file={target_url}"

    print(f"  Fetching annotation: {target_url}")
    annot_r = requests.get(target_url, timeout=120)
    annot_r.raise_for_status()

    # handle gzip
    if target_url.endswith(".gz"):
        content = gzip.decompress(annot_r.content).decode("utf-8", errors="replace")
    else:
        content = annot_r.text

    lines = content.split("\n")
    data_lines = [l for l in lines if not l.startswith(('^', '!', '#')) and l.strip()]
    print(f"  Parsed {len(data_lines)} data lines")

    annot_df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep="\t", on_bad_lines="skip")
    print(f"  Annotation shape: {annot_df.shape}, columns: {annot_df.columns.tolist()}")

    # 4. Use Groq to identify the correct columns inside the annotation file
    columns_list = annot_df.columns.tolist()
    
    column_prompt = f"""
    Given the following columns from a biological annotation file:
    {columns_list}

    1. Identify the column that maps to the index/ID column of a counts matrix (e.g., ID, Probe_ID, ID_REF).
    2. Identify the column that looks like it contains official Gene Symbols (e.g., Gene Symbol, Symbol, GENE_SYMBOL).
    3. Identify the column that contains Ensembl Gene IDs (ENSG) if a Gene Symbol is missing or as a backup (e.g., Ensembl, ENSG).

    Return your answer strictly as a JSON object with keys: "id_column", "symbol_column", and "ensg_column". If not found, use null.
    Ensure that each of the responses is solely the column name with no additional text.
    """

    col_completion = _get_client().chat.completions.create(
        messages=[
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": column_prompt}
        ],
        model="llama-3.1-8b-instant",
        response_format={"type": "json_object"}
    )

    col_data = json.loads(col_completion.choices[0].message.content)
    id_col = col_data.get("id_column")
    symbol_col = col_data.get("symbol_column")
    ensg_col = col_data.get("ensg_column")

    print(f"--> Column Mapping found by Groq: {col_data}")

    if not id_col:
        print("Could not map the ID column. Aborting mapping.")
        return counts_df

    # Determine which target column to use for replacing (Prioritize Symbol, fallback to ENSG)
    mapping_target_col = symbol_col if symbol_col and symbol_col in annot_df.columns else ensg_col

    if not mapping_target_col or mapping_target_col not in annot_df.columns:
        print("Neither Gene Symbol nor ENSG column could be reliably found.")
        return counts_df

    # 5. Map the index of counts_df using the annotation file
    if id_col not in annot_df.columns:
        print(f"  [warn] id_col '{id_col}' not found in annotation columns: {annot_df.columns.tolist()}")
        return counts_df

    if mapping_target_col not in annot_df.columns:
        print(f"  [warn] mapping_target_col '{mapping_target_col}' not found in annotation columns: {annot_df.columns.tolist()}")
        return counts_df

    mapping_dict = dict(zip(
        annot_df[id_col].astype(str),
        annot_df[mapping_target_col].astype(str),
    ))

    original_index = counts_df.index.astype(str)
    new_index = original_index.map(mapping_dict)
    counts_df.index = new_index.where(new_index.notna() & (new_index != "nan"), original_index)
    counts_df.index.name = "gene_id"

    print(f"  Successfully remapped index using '{mapping_target_col}'")
    return counts_df

def _unpack_tar(raw: bytes) -> dict[str, bytes]:
    buf = io.BytesIO(raw)
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=buf, mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if _is_tabular(name):
                f = tf.extractfile(member)
                if f:
                    members[name] = f.read()
    return members

def _collect_supp_urls(gse) -> list[dict]:
    seen: set[str] = set()
    files: list[dict] = []

    all_supps = list(gse.metadata.get("supplementary_file", []))
    for gsm in gse.gsms.values():
        all_supps.extend(gsm.metadata.get("supplementary_file", []))

    for url in all_supps:
        url = url.strip()
        if not url or url.lower() == "none" or url in seen:
            continue
        seen.add(url)
        fname = url.split("/")[-1]
        lower = fname.lower()
        is_tar = lower.endswith(".tar") or lower.endswith(".tar.gz")
        if is_tar or _is_tabular(fname):
            files.append({"url": url, "filename": fname, "is_tar": is_tar})
        else:
            print(f"  [skip] {fname}")
    
    return files


def _collect_gsm_supp_urls(gse) -> list[dict]:
    seen: set[str] = set()
    files: list[dict] = []

    for gsm_id in gse.gsms.keys():
        url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gsm_id}&targ=self&form=text&view=quick"
        try:
            r = _requests.get(url, timeout=30)
            r.raise_for_status()
        except Exception as e:
            print(f"  [warn] Could not fetch {gsm_id}: {e}")
            continue

        for line in r.text.splitlines():
            if not line.startswith("!Sample_supplementary_file"):
                continue
            val = line.split("=", 1)[-1].strip()
            if not val or val.lower() == "none" or val in seen:
                continue
            seen.add(val)
            fname = val.split("/")[-1]
            lower = fname.lower()
            is_tar = lower.endswith(".tar") or lower.endswith(".tar.gz")
            
            if is_tar or _is_tabular(fname):
                files.append({"url": val, "filename": fname, "is_tar": is_tar})
                print(f"    ✓ {gsm_id}: {fname}")

    return files


def _no_merge_build(
    url_bytes: dict[str, bytes],
    file_meta: list[dict],
) -> pd.DataFrame:
    """
    Build a count matrix from files that already contain a full 2D matrix (e.g. GSE183620 CSV/TSV files).
    Unlike _build_matrix(), this does NOT concatenate
    """

    dfs: list[pd.DataFrame] = []

    for meta in file_meta:
        url = meta["url"]
        filename = meta["filename"]

        raw = url_bytes.get(url)
        if raw is None:
            continue

        try:
            # decompress gz if needed
            if filename.endswith(".gz"):
                raw = gzip.decompress(raw)
                filename = filename[:-3]

            # delimiter detection
            if filename.endswith(".csv"):
                sep = ","
            elif filename.endswith((".tsv", ".txt")):
                sep = "\t"
            else:
                raise ValueError(f"Unsupported tabular file: {filename}")

            print(f"  Parsing matrix: {filename}")

            df = pd.read_csv(
                io.BytesIO(raw),
                sep=sep,
                index_col=0,
            )

            # clean numeric conversion
            df = df.apply(pd.to_numeric, errors="coerce")

            # standardize index name
            df.index.name = "gene_id"

            print(
                f"    ✓ Parsed matrix: "
                f"{df.shape[0]} genes x {df.shape[1]} samples"
            )

            dfs.append(df)

        except Exception as e:
            print(f"  [warn] {filename}: {e}")

    if not dfs:
        raise RuntimeError("No matrix files could be parsed.")

    if len(dfs) == 1:
        return dfs[0]

    print(f"\n  Merging {len(dfs)} matrices…")

    merged = pd.concat(dfs, axis=1, join="outer")

    print(
        f"  ✓ Final matrix: "
        f"{merged.shape[0]} genes x {merged.shape[1]} samples"
    )

    return merged

def _build_matrix(url_bytes: dict[str, bytes], file_meta: list[dict]) -> pd.DataFrame:
    # ...existing code...
    return pd.DataFrame() 

def _merge_gsm_files(url_bytes: dict[str, bytes], file_meta: list[dict]) -> pd.DataFrame:
    """Merge individual GSM supplementary files into a single matrix.
    
    Each file is treated as a full count matrix (genes x samples).
    Uses Groq to identify which numeric column represents the expression values.
    """
    dfs: list[pd.DataFrame] = []
    
    for meta in file_meta:
        url = meta["url"]
        filename = meta["filename"]
        raw = url_bytes.get(url)
        
        if raw is None:
            continue
        
        try:
            # Decompress if needed
            raw, bare = _decompress(raw, filename)
            
            # Detect separator
            sep = _detect_sep(bare)
            
            # Read the file
            df = pd.read_csv(
                io.StringIO(raw.decode("utf-8", errors="replace")),
                sep=sep, comment="#", header=0,
            )
            
            print(f"  Parsing GSM file: {filename}")
            print(f"    Shape: {df.shape}, columns: {df.columns.tolist()}")
            
            # Identify numeric columns
            num_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().all()]
            str_cols = [c for c in df.columns if c not in num_cols]
            
            if not num_cols:
                print(f"    [warn] No numeric columns found, skipping")
                continue
            
            # Use Groq to pick the correct count column
            if len(num_cols) > 1:
                prompt = f"""
Given a GSM supplementary file with these numeric columns:
{json.dumps(num_cols)}

Which one contains the primary expression values (counts, CPM, FPKM, TPM)?
Return ONLY the column name, nothing else.
"""
                resp = _get_client().chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a bioinformatics expert. Return only the column name."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0, max_tokens=50,
                )
                chosen = resp.choices[0].message.content.strip().strip('"').strip("'")
                count_col = next((c for c in num_cols if c.lower() == chosen.lower()), num_cols[0])
            else:
                count_col = num_cols[0]
            
            print(f"    Using column: {count_col}")
            
            # Use first string column as gene ID, or index if none
            if str_cols:
                df = df.set_index(str_cols[0])
            
            # Extract only the count column and convert to numeric
            df = df[[count_col]].apply(pd.to_numeric, errors="coerce")
            df.columns = [filename.split('.')[0]]  # Use filename stem as sample name
            
            print(f"    ✓ Parsed: {df.shape[0]} genes")
            dfs.append(df)
            
        except Exception as e:
            print(f"  [warn] {filename}: {e}")
    
    if not dfs:
        raise RuntimeError("No count data could be parsed from GSM files.")
    
    # Merge all DataFrames on gene index
    print(f"\n  Merging {len(dfs)} GSM files…")
    merged = pd.concat(dfs, axis=1, join="outer")
    merged.index.name = "gene_id"
    
    print(f"  ✓ Merged matrix: {merged.shape[0]} genes × {merged.shape[1]} samples")
    return merged 

def _get_geo_metadata(accession: str, cache_dir: Path):
    soft_path = cache_dir / f"{accession}_family.soft.gz"

    if not soft_path.exists():
        prefix = accession[:-3] + "nnn" 
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{accession}/soft/{accession}_family.soft.gz"
        url = url.replace("ftp://", "https://", 1)
        print(f"  Downloading SOFT: {url}")
        r = _requests.get(url, timeout=300)
        r.raise_for_status()
        soft_path.write_bytes(r.content)
        print(f"  Saved to {soft_path}")
    else:
        print(f"  Using cached SOFT: {soft_path}")

    return GEOparse.get_GEO(filepath=str(soft_path), silent=True)

def fetch_and_normalize_ncbi_gen(
    accession: str,
    file_meta: list[dict],
    geo_cache_dir: str | Path = "./geo_cache",
) -> pd.DataFrame:
    _download_raw(accession, file_meta, geo_cache_dir)
    url_bytes = _download_all([m["url"] for m in file_meta])
    if (file_meta[0]["is_tar"]):
        counts_df = _build_matrix(url_bytes, file_meta)
    else:
        counts_df = _no_merge_build(url_bytes, file_meta)
    counts_df = _annotate_raw(accession, file_meta, counts_df)
    return counts_df

def fetch_and_normalize(
    accession: str,
    geo_cache_dir: str | Path = "./geo_cache",
    save_output: bool = False,
    output_dir: str | Path = "./geo_output",
) -> GseResult:
    geo_cache_dir = Path(geo_cache_dir)
    geo_cache_dir.mkdir(parents=True, exist_ok=True)
    accession = accession.strip().upper()

    print(f"Fetching GEO metadata: {accession}")
    geo = _get_geo_metadata(accession, geo_cache_dir)

    if accession.startswith("GSE"):
        gsm_dict = geo.gsms
    elif accession.startswith("GSM"):
        gsm_dict = {accession: geo}
    else:
        raise ValueError(f"Unknown accession type: {accession!r}")

    first_gsm = next(iter(gsm_dict.values()))
    dp_text   = "\n".join(first_gsm.metadata.get("data_processing", []))

    if dp_text.strip():
        print("Classifying normalization type (once for the whole dataset)…")
        clf        = classify_normalization(dp_text)
        norm_type  = clf["normalization_type"]
        confidence = clf["confidence"]
        reasoning  = clf["reasoning"]
        print(f"  → {norm_type}  [{confidence}]  {reasoning}")
    else:
        norm_type, confidence, reasoning = "unknown", "low", "No data_processing field"
        print("  [warn] No data_processing metadata found.")

    if accession.startswith("GSE"):
        file_meta = _collect_supp_urls(geo)
    else:
        file_meta = []
        for url in first_gsm.metadata.get("supplementary_file", []):
            url = url.strip()
            if not url or url.lower() == "none":
                continue
            fname  = url.split("/")[-1]
            lower  = fname.lower()
            is_tar = lower.endswith(".tar") or lower.endswith(".tar.gz")
            if is_tar or _is_tabular(fname):
                file_meta.append({"url": url, "filename": fname, "is_tar": is_tar})

    if not file_meta:
        print("[skip] No supported supplementary files found.")
        return GseResult(accession=accession, normalization_type=norm_type,
                         effective_norm=norm_type, confidence=confidence,
                         reasoning=reasoning)

    print(f"\nFound {len(file_meta)} supplementary file(s):")
    for m in file_meta:
        print(f"  {'[TAR]' if m['is_tar'] else '     '} {m['filename']}")

    # ── download ──────────────────────────────────────────────────────────────
    if norm_type == "cpm" or norm_type == "raw_counts":
        found = _download_raw(accession, file_meta, geo_cache_dir)
        if found == 0:
            # Check if we only have TAR files
            if file_meta and all(m["is_tar"] for m in file_meta):
                print(f"  Found only TAR files. Checking if GSMs have individual supplementary files…")
                gsm_files = _collect_gsm_supp_urls(geo)
                if gsm_files and not all(f["is_tar"] for f in gsm_files):
                    print(f"  ✓ Found {len(gsm_files)} individual GSM supplementary files")
                    file_meta = gsm_files
                    url_bytes = _download_all([m["url"] for m in file_meta])
                    counts_df = _merge_gsm_files(url_bytes, file_meta)
                else:
                    url_bytes = _download_all([m["url"] for m in file_meta])
                    counts_df = _build_matrix(url_bytes, file_meta)
            else: 
                url_bytes = _download_all([m["url"] for m in file_meta])
                if file_meta and file_meta[0]["is_tar"]:
                    counts_df = _build_matrix(url_bytes, file_meta)
                else:
                    counts_df = _no_merge_build(url_bytes, file_meta)
        else:
            """# Check if we only have TAR files
            if file_meta and all(m["is_tar"] for m in file_meta):
                print(f"  Found only TAR files. Checking if GSMs have individual supplementary files…")
                gsm_files = _collect_gsm_supp_urls(geo)
                if gsm_files and not all(f["is_tar"] for f in gsm_files):
                    print(f"  ✓ Found {len(gsm_files)} individual GSM supplementary files")
                    file_meta = gsm_files
                    url_bytes = _download_all([m["url"] for m in file_meta])
                    counts_df = _merge_gsm_files(url_bytes, file_meta)
                else:
                    url_bytes = _download_all([m["url"] for m in file_meta])
                    counts_df = _build_matrix(url_bytes, file_meta)
            else:
                counts_df = fetch_and_normalize_ncbi_gen(accession, file_meta, geo_cache_dir)
                # Found raw count files, reassign normalization type
                if file_meta and any("raw" in m["filename"].lower() for m in file_meta):
                    print(f"  [reassign] Found raw count files; updating norm_type: {norm_type} → raw_counts")
                    norm_type = "raw_counts"
            """
            counts_df = fetch_and_normalize_ncbi_gen(accession, file_meta, geo_cache_dir)
            # Found raw count files, reassign normalization type
            if file_meta and any("raw" in m["filename"].lower() for m in file_meta):
                print(f"  [reassign] Found raw count files; updating norm_type: {norm_type} → raw_counts")
                norm_type = "raw_counts"

    else:
        # Check if we only have TAR files
        """if file_meta and all(m["is_tar"] for m in file_meta):
            print(f"  Found only TAR files. Checking if GSMs have individual supplementary files…")
            gsm_files = _collect_gsm_supp_urls(geo)
            if gsm_files and not all(f["is_tar"] for f in gsm_files):
                print(f"  ✓ Found {len(gsm_files)} individual GSM supplementary files")
                file_meta = gsm_files"""
        
        counts_df = fetch_and_normalize_ncbi_gen(accession, file_meta, geo_cache_dir)
        # Found raw count files, reassign normalization type
        if file_meta and any("raw" in m["filename"].lower() for m in file_meta):
            print(f"  [reassign] Found raw count files; updating norm_type: {norm_type} → raw_counts")
            norm_type = "raw_counts"

    # ── convert to CPM then log2(CPM+1) ──────────────────────────────────────
    print(f"\n  Converting '{norm_type}' → CPM → log2(CPM + 1)…")
    cpm_df, effective_norm = convert_to_cpm(counts_df, norm_type)
    log2cpm_df = to_log2cpm(cpm_df)
    print(f"  ✓ effective_norm={effective_norm!r}  "
          f"log2(CPM+1) range: {log2cpm_df.min().min():.2f} – {log2cpm_df.max().max():.2f}")

    result = GseResult(
        accession=accession,
        normalization_type=norm_type,
        effective_norm=effective_norm,
        confidence=confidence,
        reasoning=reasoning,
        counts_df=counts_df,
        cpm_df=cpm_df,
        log2cpm_df=log2cpm_df,
    )

    if save_output:
        result.save(output_dir)

    return result