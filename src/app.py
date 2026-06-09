import asyncio
import io
import json
import os

import GEOparse
import httpx
import numpy as np
import pandas as pd
import requests
import streamlit as st
from transformer.gene_symbol import *
import geo_rnaseq_normalizer as geo_norm
from transformers import pipeline

st.set_page_config(page_title="GEOlogist", page_icon="🧬")
st.title("GEOlogist 🧬")

gse_id = st.text_input("Enter GEO accession:", placeholder="e.g. GSE183620")

if not gse_id:
    st.stop()

gse_id = gse_id.strip().upper()

@st.cache_data(show_spinner="Fetching GEO metadata…")
def fetch_metadata(gse_id: str) -> dict:
    try:
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=gds&term={gse_id}[Accession]&retmode=json"
        )
        print(requests.get(search_url).json()["esearchresult"])
        uid = requests.get(search_url).json()["esearchresult"]["idlist"][0]
        sum_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
            f"?db=gds&id={uid}&retmode=json"
        )
        summary = requests.get(sum_url).json()["result"][uid]

        raw_gpl  = summary.get("gpl", "")
        gpl_ids  = [f"GPL{g.strip()}" for g in raw_gpl.split(";") if g.strip()]

        gsm_list, gsm_to_gpl = [], {}
        for s in summary.get("samples", []):
            acc = s["accession"]
            gsm_list.append(acc)
            gsm_to_gpl[acc] = f"GPL{s['gpl']}" if "gpl" in s else (gpl_ids[0] if gpl_ids else "")

        study_type = (
            "Microarray" if "array" in summary.get("gdstype", "").lower()
            else "RNA-seq"
        )
        return {
            "title":      summary.get("title", gse_id),
            "type":       study_type,
            "gsm_ids":    gsm_list,
            "gpl_ids":    gpl_ids,
            "gsm_to_gpl": gsm_to_gpl,
            "taxon":      summary.get("taxon", ""),
            "error":      None,
        }
    except Exception as e:
        return {"error": str(e)}


meta = fetch_metadata(gse_id)

if meta.get("error"):
    st.error(f"Failed to load metadata: {meta['error']}")
    st.stop()

st.subheader(meta["title"])
st.caption(
    f"Type: **{meta['type']}** · "
    f"Samples: **{len(meta['gsm_ids'])}** · "
    f"Taxon: {meta['taxon']}"
)


async def _fetch_single_gsm(
    client: httpx.AsyncClient,
    gsm_id: str,
    progress_bar,
    progress_text,
    idx: int,
    total: int,
) -> pd.DataFrame | None:
    url = (
        f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
        f"?acc={gsm_id}&targ=1&form=text&view=data"
    )
    try:
        resp = await client.get(url, timeout=60.0)
        if "!sample_table_begin" in resp.text:
            table_part = (
                resp.text
                .split("!sample_table_begin")[1]
                .split("!sample_table_end")[0]
            )
            df = pd.read_csv(io.StringIO(table_part.strip()), sep="\t")
            if "VALUE" in df.columns:
                df = df.set_index(df.columns[0])[["VALUE"]]
                df.columns = [gsm_id]
                progress_bar.progress((idx + 1) / total)
                progress_text.text(f"Downloaded {gsm_id} ({idx + 1}/{total})")
                return df
    except Exception:
        pass
    return None


async def _inhale_all_gsms(gsm_ids: list[str]) -> list[pd.DataFrame]:
    limits = httpx.Limits(max_keepalive_connections=40, max_connections=50)
    progress_text = st.empty()
    progress_bar  = st.progress(0)
    total = len(gsm_ids)

    async with httpx.AsyncClient(limits=limits, verify=False) as client:
        tasks = [
            _fetch_single_gsm(client, gsm_id, progress_bar, progress_text, i, total)
            for i, gsm_id in enumerate(gsm_ids)
        ]
        results = await asyncio.gather(*tasks)

    progress_text.empty()
    progress_bar.empty()
    return [r for r in results if r is not None]


def _fetch_gpl_tables(gpl_ids: list[str]) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for gpl_id in gpl_ids:
        with st.spinner(f"Downloading platform annotation {gpl_id}…"):
            gpl = GEOparse.get_GEO(geo=gpl_id, destdir="./geo_cache", silent=True)
            tables[gpl_id] = gpl.table
    return tables


def _annotate_matrix(df: pd.DataFrame, gpl_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = df.reset_index()
    df.rename(columns={df.columns[0]: "_probe_id"}, inplace=True)

    probe_ids = [str(pid).strip() for pid in df["_probe_id"]]
    looks_like_symbols = all(
        not (
            pid.isdigit() or
            pid.startswith("ENSG") or pid.startswith("ENST") or
            pid.startswith("AFFX") or pid.startswith("ILMN") or
            pid.startswith("A_") or pid.endswith("_at")
        )
        for pid in probe_ids[:min(100, len(probe_ids))]
    )

    if looks_like_symbols:
        df.insert(0, "Name", df["_probe_id"])
        df = df.drop("_probe_id", axis=1)
        return df

    NULL_VALUES = {"---", "na", "", "null", "nan"}

    gpl_table, matching_col, symbol_col = get_gene_symbol_column(
        gse_id, df, probe_ids=df["_probe_id"]
    )
    print(f"  → matching_col='{matching_col}'  symbol_col='{symbol_col}'")

    master_mapping: dict[str, str] = {}
    if symbol_col in gpl_table.columns and matching_col in gpl_table.columns:
        for probe, sym in zip(
            gpl_table[matching_col].astype(str).str.strip(),
            gpl_table[symbol_col].astype(str).str.strip(),
        ):
            if probe and sym and sym.lower() not in NULL_VALUES:
                if "//" in sym:
                    sym = sym.split("//")[1].strip()
                master_mapping[probe] = sym

    df["Name"] = df["_probe_id"].astype(str).str.strip().map(master_mapping)

    before = len(df)
    df = df.dropna(subset=["Name"])
    after = len(df)
    if before != after:
        print(f"dropped {before - after} unmapped rows ({after} remaining)")

    df = df.drop("_probe_id", axis=1)
    print(f"Remapped using '{matching_col}' → '{symbol_col}'")
    return df


def _check_log_status(gsm_id: str) -> bool:
    try:
        url = (
            f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
            f"?acc={gsm_id}&targ=self&view=data&form=text&lines=0"
        )
        res = requests.get(url, timeout=5).text
        return "log" in res.lower()
    except Exception:
        return False

_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "zero-shot-classification",
            model="cross-encoder/nli-deberta-v3-small" 
        )
    return _classifier
    
label_dict = {
    "raw unnormalized counts": "raw_counts",
    "cpm normalized": "cpm",
    "RPKM or FPKM normalized": "rpkm_fpkm",
    "TPM normalized": "tpm",
    "geometric mean": "geometric_mean",
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
    "geometric mean",
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

def to_cpm(df: pd.DataFrame) -> pd.DataFrame:
    """Divide each column by its sum and scale to 1e6."""
    # Ensure only numeric columns
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    col_sums = numeric_df.sum(axis=0).replace(0, np.nan)
    return numeric_df.divide(col_sums, axis=1) * 1e6

def fetch_microarray_matrix(meta: dict):
    gpl_tables = _fetch_gpl_tables(meta["gpl_ids"])
    geo = GEOparse.get_GEO(gse_id)
    if gse_id.startswith("GSE"):
        gsm_dict = geo.gsms
    else:
        gsm_dict = {gse_id: geo}
 
    first_gsm = next(iter(gsm_dict.values()))
    dp_text = "\n".join(first_gsm.metadata.get("data_processing", []))
    if dp_text.strip():
        print("Classifying normalization type…")
        clf = classify_normalization(dp_text)
        norm_type = clf["normalization_type"]
        confidence = clf["confidence"]
        reasoning = clf["reasoning"]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    all_dfs = loop.run_until_complete(_inhale_all_gsms(meta["gsm_ids"]))

    if not all_dfs:
        return None

    cleaned = []
    for df in all_dfs:
        df = df[~df.index.astype(str).str.contains("AFFX")]
        if "N/A" in df.index:
            df = df.drop("N/A")
        cleaned.append(df)

    st.write("Merging samples…")
    final_df = pd.concat(cleaned, axis=1, join="outer")
    final_df = _annotate_matrix(final_df, gpl_tables)

    # log2(x + 1) normalisation if not already log-transformed
    is_log = _check_log_status(meta["gsm_ids"][0]) if meta["gsm_ids"] else False
    if not is_log and norm_type == "raw":
        final_df = to_cpm(final_df)
        nums = final_df.select_dtypes(include=[np.number]).columns
        final_df[nums] = np.log2(final_df[nums].astype(float) + 1)
        norm_type = "log2"
    elif not is_log and norm_type == "cpm":
        nums = final_df.select_dtypes(include=[np.number]).columns
        final_df[nums] = np.log2(final_df[nums].astype(float) + 1)
        norm_type = f"log2({norm_type})"

    return final_df, norm_type

def fetch_rnaseq_matrix(gse_id: str):
    try:
        result = geo_norm.fetch_and_normalize(
            accession=gse_id,
            geo_cache_dir="./geo_cache",
            save_output=False,
        )
    except Exception as e:
        st.error(f"geo_rnaseq_normalizer error: {e}")
        return None

    if result.log2cpm_df is None:
        st.error("No matrix could be built from the supplementary files.")
        return None

    df = result.log2cpm_df.copy()
    df.index.name = None
    df.insert(0, "Name", df.index)
    df = df.reset_index(drop=True)

    if result.fpkm_conversion_failed:
        st.warning(
            f"⚠️ **FPKM→CPM conversion failed** — gene lengths could not be "
            f"fetched from Ensembl BioMart.  \n"
            f"The original **FPKM values** are returned as-is (log2 applied).  \n"
            f"Results are **not comparable across samples** without proper length normalisation."
        )
    else:
        st.info(
            f"Original normalization: **{result.normalization_type}** "
            f"[{result.confidence}] — converted to **log2(CPM + 1)**  \n"
            f"_{result.reasoning}_"
        )
    return df, result.effective_norm

if st.button("Fetch & Build Matrix", type="primary"):

    with st.status("Running pipeline…", expanded=True) as status:

        if meta["type"] == "Microarray":
            st.write("Running microarray pipeline (async GSM fetch + GPL annotation)…")
            combined_df, effective_normalization = fetch_microarray_matrix(meta)
        else:
            st.write(
                f"Running RNA-seq pipeline for **{gse_id}** "
                f"(GEO supplementary files → CPM → log2(CPM+1))…"
            )
            combined_df, effective_normalization = fetch_rnaseq_matrix(gse_id)

        if combined_df is None:
            status.update(label="Pipeline failed.", state="error")
            st.stop()

        status.update(label="Done!", state="complete")

    n_genes   = combined_df.shape[0]
    n_samples = combined_df.shape[1] - (1 if "Name" in combined_df.columns else 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Genes",   f"{n_genes:,}")
    c2.metric("Samples", n_samples)
    c3.metric("Normalization", str(effective_normalization))

    st.dataframe(
        combined_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Name": st.column_config.TextColumn("Gene", pinned=True, width="small"),
        },
    )

    tsv = combined_df.to_csv(sep="\t", index=False)
    st.download_button(
        label="⬇ Download as .txt (tab-separated)",
        data=tsv,
        file_name=f"{gse_id}_{effective_normalization}.txt",
        mime="text/plain",
    )