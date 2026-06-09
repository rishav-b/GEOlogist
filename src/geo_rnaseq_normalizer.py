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
from transformer.gene_symbol import *
import time
import warnings
import numpy as np
import GEOparse
import pandas as pd
import requests as _requests

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

from transformers import pipeline

_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small" 
        )
    return _classifier

GROQ_MODEL = "llama-3.3-70b-versatile"
TABULAR_EXTS = {".txt", ".csv", ".tsv", ".tab", ".xlsx", ".xls"}
_CHUNK   = 64 * 1024 * 1024   # 64 MB read buffer for faster downloads
_WORKERS = 64

#normalization

def to_cpm(df: pd.DataFrame) -> pd.DataFrame:
    """Divide each column by its sum and scale to 1e6."""
    # Ensure only numeric columns
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    col_sums = numeric_df.sum(axis=0).replace(0, np.nan)
    return numeric_df.divide(col_sums, axis=1) * 1e6

def _to_log2cpm(cpm: pd.DataFrame) -> pd.DataFrame:
    """log2(CPM + 1)."""
    return np.log2(cpm + 1)

def _detect_log_base(df: pd.DataFrame) -> str:
    """Heuristic: max < 35 → log2, else natural log."""
    return "log2" if df.max().max() < 35 else "loge"

def convert_to_cpm(df: pd.DataFrame, norm_type: str) -> tuple[pd.DataFrame, str]:
    nt = norm_type.lower()

    if nt == "raw_counts":
        print("  [convert] raw_counts → CPM")
        return to_cpm(df), "cpm"

    elif nt == "rpkm_fpkm":
        print("  leaving as rpkm/fpkm")
        return df, "rpkm_fpkm"

    elif nt == "tpm":
        print("  leaving as tpm")
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
class FileMeta:
    """Metadata for one downloadable supplementary file."""
    url:      str
    filename: str
    is_tar:   bool = False
    ncbi_data: bool = False
 
    @property
    def is_tabular(self) -> bool:
        return _is_tabular(self.filename)

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
    

#normlization


label_dict = {
    "raw unnormalized counts": "raw_counts",
    "cpm normalized": "cpm",
    "RPKM or FPKM normalized": "rpkm_fpkm",
    "TPM normalized": "tpm",
    "Geometric mean normalized": "geometric_mean",
    "log transformed expression":  "log_transformed",
    "VST or rlog variance stabilized": "vst_rlog",
    "quantile normalized":    "quantile",
    "TMM or RLE normalized":  "tmm_rle",
    "RMA normalized microarray expression": "microarray_rma",
    "fold change between conditions": "fold_change",
    "z-score normalized":  "z_score",
}

candidate_labels = [    "raw unnormalized counts",
    "cpm normalized",
    "RPKM or FPKM normalized",
    "TPM normalized",
    "Geometric mean normalized",
    "log transformed expression",
    "VST or rlog variance stabilized",
    "quantile normalized",
    "TMM or RLE normalized",
    "RMA normalized microarray expression",
    "fold change between conditions",
    "z-score normalized"]

def classify_normalization(dp_text: str) -> dict:
    if not dp_text or not dp_text.strip():
        return {"normalization_type": "unknown", "confidence": "low", "reasoning": "Empty input"}

    result = _get_classifier()(
        dp_text,
        candidate_labels=candidate_labels,
        hypothesis_template="The supplementary gene expression data contains {}.",
    )
    
    print(result)

    top_label = result["labels"][0]
    top_score = result["scores"][0]
    norm_type = label_dict[top_label]
    confidence = "high" if top_score > 0.75 else "medium" if top_score > 0.5 else "low"

    return {
        "normalization_type": norm_type,
        "confidence": confidence,
        "reasoning": f"{top_label} (score: {top_score:.2f})",
    }

#bytes download

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

#simple helper functions

def _is_tabular(name: str) -> bool:
    lower = name.lower()
    return any(lower.endswith(e) or lower.endswith(e + ".gz") for e in TABULAR_EXTS)


def _decompress(raw: bytes, name: str) -> tuple[bytes, str]:
    if name.lower().endswith(".gz"):
        return gzip.decompress(raw), name[:-3]
    return raw, name


def _detect_sep(name: str) -> str:
    return "," if name.lower().endswith(".csv") else "\t"


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

#tested
def _scrape_geo_download_page(accession):
    page_url = f"https://www.ncbi.nlm.nih.gov/geo/download/?acc={accession}"
    print(f"  Scraping GEO download page: {page_url}")
 
    r = requests.get(page_url, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
 
    seen = set()
    files = []
 
    for a in soup.find_all("a", href=True):
        href = a["href"]
 
        if "file=" in href:
            filename = href.split("file=")[-1].split("&")[0]
        elif "suppl/" in href:
            filename = href.split("suppl/")[-1]
        else:
            continue

        if not filename or filename in seen:
            continue
        seen.add(filename)
 
        full_url = urljoin(page_url, href)
        lower    = filename.lower()
        is_tar   = lower.endswith(".tar") or lower.endswith(".tar.gz")
        ncbi_data = "file=" in href
 
        if not (is_tar or _is_tabular(filename)):
            print(f"    [skip] {filename}")
            continue
        
        files.append(FileMeta(url=full_url, filename=filename, is_tar=is_tar, ncbi_data=ncbi_data))
        print(f"    found: {filename}")
 
    print(f"{len(files)} candidates on download page")
    return files

def select_download_urls(candidates, accession, clf):
    submitter_candidates = []
    for c in candidates:
        if c.ncbi_data and "raw" in c.filename:
            return [c], clf
        elif not c.ncbi_data:
            submitter_candidates.append(c)
    if len(submitter_candidates) == 1:
        return submitter_candidates, clf
    else:
        for s in submitter_candidates:
            if (any(x in s.filename.lower() for x in ["raw", "counts", "rsem", "count"])):
                return [s], {"normalization_type": "raw_counts", "confidence": "high", "reasoning": "Contains raw counts keywords"}
        
        norm_type  = clf.get("normalization_type", "unknown")
        hypothesis = f"This file contains {norm_type.replace('_', ' ')} gene expression data."
        filenames  = [s.filename for s in submitter_candidates]

        result = _get_classifier()(
            filenames,
            candidate_labels=[norm_type],
            hypothesis_template=hypothesis,
            multi_label=True,
        )

        # _get_classifier() returns a list when given a list of sequences
        if isinstance(result, dict):
            result = [result]

        scored = sorted(
            zip(submitter_candidates, result),
            key=lambda x: x[1]["scores"][0],
            reverse=True,
        )

        best_candidate, best_result = scored[0]
        best_score = best_result["scores"][0]
        confidence = "high" if best_score > 0.75 else "medium" if best_score > 0.5 else "low"

        updated_clf = {
            "normalization_type": norm_type,
            "confidence":         confidence,
            "reasoning":          f"Transformer selected '{best_candidate.filename}' "
                                  f"(score: {best_score:.2f}) as most likely {norm_type}",
        }

        return [best_candidate], updated_clf

    

#constructing the dataframe

def _parse_tabular_series(raw: bytes, filename: str) -> tuple[pd.Series, str]:
    sep = _detect_sep(filename)
    df  = pd.read_csv(
        io.StringIO(raw.decode("utf-8", errors="replace")),
        sep=sep, comment="#", header=0,
    )

    mask = df.iloc[:, 0].astype(str).str.startswith("__")
    df = df[~mask.astype(bool)]

    num_cols = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().all()]
    str_cols = [c for c in df.columns if c not in num_cols]

    # Gene index from first non-numeric column
    gene_index = df[str_cols[0]].astype(str) if str_cols else df.index.astype(str)

    if len(num_cols) == 0:
        raise ValueError(f"No numeric columns in {filename}")
    else:
        count_col = num_cols[0]

    s = pd.to_numeric(df[count_col], errors="coerce")
    s.index = gene_index

    stem = re.sub(r"\.gz$", "", filename, flags=re.IGNORECASE)
    stem = re.sub(r"\.(csv|tsv|txt|tab)$", "", stem, flags=re.IGNORECASE)
    return s, stem

def _build_matrix_from_tar(url_bytes, file_meta):
    series_list = []

    for meta in file_meta:
        raw = url_bytes.get(meta.url)
        if raw is None:
            continue
        members = _unpack_tar(raw)
        for mname, mbytes in members.items():
            mbytes, bare = _decompress(mbytes, mname)
            try:
                s, sname = _parse_tabular_series(mbytes, bare)
                s.name   = sname
                series_list.append(s)
            except Exception as e:
                print(f"    [warn] {mname}: {e}")
 
    if not series_list:
        raise RuntimeError("No count data found inside TAR archive(s).")
 
    print(f"\n  Merging {len(series_list)} sample series from TAR…")
    df = pd.concat(series_list, axis=1, join="outer")
    df = df.apply(pd.to_numeric, errors="coerce")
    df.index.name = "gene_id"
    print(df)
    print(f"  ✓ Matrix: {df.shape[0]} genes × {df.shape[1]} samples")
    return df

def _build_matrix_from_flat_files(url_bytes, file_meta):
    dfs = []

    for meta in file_meta:
        raw = url_bytes.get(meta.url)
        if raw is None:
            continue
        filename = meta.filename
        try:
            if filename.endswith(".gz"):
                raw = gzip.decompress(raw)
                filename = filename[:-3]
            sep = _detect_sep(filename)

            lower = filename.lower()
            if lower.endswith(".xls") or lower.endswith(".xlsx"):
                df = pd.read_excel(io.BytesIO(raw), index_col=0)
            else:
                df = pd.read_csv(
                    io.BytesIO(raw), sep=_detect_sep(filename),
                    index_col=0, engine="python", on_bad_lines="skip",
                )
            df = df.apply(pd.to_numeric, errors = "coerce")
            df.index.name = "gene_id"

            df = df[~df.index.astype(str).str.startswith("__")]

            frac_numeric = df.notna().mean()
            df = df.loc[:, frac_numeric > 0.9]

            dfs.append(df)
        except Exception as e:
            print(f"Error: {e}")

    if not dfs:
        raise RuntimeError("No matrix files parseable")
    
    if len(dfs) == 1:
        return dfs[0]
    
    print("Merging...")
    merged = pd.concat(dfs, axis=1, join = "outer")
    return merged
    

def _fetch_counts_df(accession, file_meta, url_bytes):

    tar_files  = [m for m in file_meta if m.is_tar]
    flat_files = [m for m in file_meta if not m.is_tar]
 
    if tar_files:
        return _build_matrix_from_tar(url_bytes, tar_files)
    elif flat_files:
        return _build_matrix_from_flat_files(url_bytes, flat_files)
    else:
        raise RuntimeError("No usable files in file_meta.")

def _annotate_counts(accession, selected, counts_df):
    if selected[0].ncbi_data:
        annot_url = "https://www.ncbi.nlm.nih.gov/geo/download/?format=file&type=rnaseq_counts&file=Human.GRCh38.p13.annot.tsv.gz"
        url_bytes = _download_all([annot_url])
        raw = url_bytes[annot_url]
        filename = annot_url.split("file=")[-1]

        if filename.endswith(".gz"):
            raw = gzip.decompress(raw)
            filename = filename[:-3]

        annot_df = pd.read_csv(
            io.BytesIO(raw), sep=_detect_sep(filename),
            engine="python", on_bad_lines="skip",
        )
        
        id_col = "GeneID"
        symbol_col = "Symbol"
    else:
        try:
            annot_df, id_col, symbol_col = get_gene_symbol_column(accession, counts_df)
        except:
            return counts_df

    mapping     = dict(zip(annot_df[id_col].astype(str), annot_df[symbol_col].astype(str)))
    orig_index  = counts_df.index.astype(str)
    new_index   = orig_index.map(mapping)
    counts_df.index = new_index.where(
        new_index.notna() & (new_index != "nan"), orig_index
    )
    counts_df.index.name = "gene_id"
    print(f"  ✓ Index remapped using '{symbol_col}'")
    return counts_df

def _get_geo_metadata(accession: str, cache_dir: Path):
    soft_path = cache_dir / f"{accession}_family.soft.gz"
 
    if not soft_path.exists():
        prefix = accession[:-3] + "nnn"
        url    = (
            f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/"
            f"{accession}/soft/{accession}_family.soft.gz"
        ).replace("ftp://", "https://", 1)
        print(f"  Downloading SOFT: {url}")
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        soft_path.write_bytes(r.content)
        print(f"  Saved to {soft_path}")
    else:
        print(f"  Using cached SOFT: {soft_path}")
 
    return GEOparse.get_GEO(filepath=str(soft_path), silent=True)

def _read_dp_text(geo, accession):
    if accession.startswith("GSE"):
        gsm_dict = geo.gsms
    else:
        gsm_dict = {accession: geo}
 
    first_gsm = next(iter(gsm_dict.values()))
    return "\n".join(first_gsm.metadata.get("data_processing", []))

def fetch_and_normalize(
    accession:     str,
    geo_cache_dir: str | Path = "./geo_cache",
    save_output:   bool       = False,
    output_dir:    str | Path = "./geo_output",
) -> GseResult:

    geo_cache_dir = Path(geo_cache_dir)
    geo_cache_dir.mkdir(parents=True, exist_ok=True)
    accession = accession.strip().upper()
 
    print(f"\n{'═'*60}")
    print(f"  GEO accession: {accession}")
    print(f"{'═'*60}")
    geo      = _get_geo_metadata(accession, geo_cache_dir)
    dp_text  = _read_dp_text(geo, accession)
 
    if dp_text.strip():
        print("Classifying normalization type…")
        clf        = classify_normalization(dp_text)
        norm_type  = clf["normalization_type"]
        confidence = clf["confidence"]
        reasoning  = clf["reasoning"]
        print(f"  → {norm_type}  [{confidence}]  {reasoning}")
    else:
        norm_type, confidence, reasoning = "unknown", "low", "No data_processing field"
 
    candidates = _scrape_geo_download_page(accession)
    if not candidates:
        return GseResult(
            accession=accession, normalization_type=norm_type,
            effective_norm=norm_type, confidence=confidence, reasoning=reasoning,
        )
 
    selected, clf = select_download_urls(candidates, accession, clf)
    if not selected:
        return GseResult(
            accession=accession, normalization_type=norm_type,
            effective_norm=norm_type, confidence=confidence, reasoning=reasoning,
        )
    norm_type  = clf["normalization_type"]
    confidence = clf["confidence"]
    reasoning  = clf["reasoning"]
 
    print(f"\n  Selected {len(selected)} file(s) to download:")
    for m in selected:
        print(f"    {'[TAR]' if m.is_tar else '     '} {m.filename}")
 
    url_bytes = _download_all([m.url for m in selected])

    print("\n  Building count matrix…")
    counts_df = _fetch_counts_df(accession, selected, url_bytes)
 
    counts_df = _annotate_counts(accession, selected, counts_df)
 
    print(f"\n  Converting '{norm_type}' → CPM → log2(CPM+1)…")
    cpm_df, effective_norm = convert_to_cpm(counts_df, norm_type)
    if effective_norm == "cpm":
        log2cpm_df = _to_log2cpm(cpm_df)
        effective_norm = "log2(CPM+1)"
    else:
        log2cpm_df = cpm_df

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