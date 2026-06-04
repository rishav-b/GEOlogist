"""
app.py  –  GEOlogist Streamlit dashboard
=========================================
Fetches any GEO study (RNA-seq or microarray), builds an expression matrix,
and normalises to log2(CPM + 1) for RNA-seq.

RNA-seq pipeline  : geo_rnaseq_normalizer  (GEO supplementary files)
Microarray pipeline: async GSM VALUE table fetch + GPL annotation + log2

Run:
    streamlit run app.py
"""

import asyncio
import io
import os

import GEOparse
import httpx
import numpy as np
import pandas as pd
import requests
import streamlit as st

import geo_rnaseq_normalizer as geo_norm

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="GEOlogist", page_icon="🧬")
st.title("GEOlogist 🧬")

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – Groq API key (required before any fetch)
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuration")
    groq_key = st.text_input(
        "Groq API key",
        type="password",
        placeholder="gsk_...",
        help="Free key at console.groq.com — required for normalization classification.",
        value=os.environ.get("GROQ_API_KEY", ""),
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key
        geo_norm._groq_client = None   # re-init if key changed

if not groq_key:
    st.info(
        "👈 Enter your **Groq API key** in the sidebar to get started.\n\n"
        "Get a free key (no credit card) at **console.groq.com → API Keys**."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# GSE input
# ─────────────────────────────────────────────────────────────────────────────

gse_id = st.text_input("Enter GEO accession:", placeholder="e.g. GSE183620")

if not gse_id:
    st.stop()

gse_id = gse_id.strip().upper()

# ─────────────────────────────────────────────────────────────────────────────
# Metadata fetch
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching GEO metadata…")
def fetch_metadata(gse_id: str) -> dict:
    try:
        search_url = (
            f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            f"?db=gds&term={gse_id}[Accession]&retmode=json"
        )
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

# ─────────────────────────────────────────────────────────────────────────────
# Microarray helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    master_mapping: dict[str, str] = {}
    for gpl_table in gpl_tables.values():
        id_col = next(
            (c for c in gpl_table.columns if c.upper() in ("ID", "ID_REF")),
            gpl_table.columns[0],
        )
        symbol_col = next(
            (c for c in gpl_table.columns if "symbol" in c.lower()), None
        )
        if symbol_col:
            for probe, sym in zip(
                gpl_table[id_col].astype(str).str.strip(),
                gpl_table[symbol_col].astype(str).str.strip(),
            ):
                master_mapping[probe] = sym

    df.insert(0, "Name", [master_mapping.get(str(idx).strip(), idx) for idx in df.index])
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


def fetch_microarray_matrix(meta: dict) -> pd.DataFrame | None:
    gpl_tables = _fetch_gpl_tables(meta["gpl_ids"])

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
    if not is_log:
        nums = final_df.select_dtypes(include=[np.number]).columns
        final_df[nums] = np.log2(final_df[nums].astype(float) + 1)

    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# RNA-seq pipeline via geo_rnaseq_normalizer
# ─────────────────────────────────────────────────────────────────────────────

def fetch_rnaseq_matrix(gse_id: str) -> pd.DataFrame | None:
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
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Fetch button
# ─────────────────────────────────────────────────────────────────────────────

if st.button("Fetch & Build Matrix", type="primary"):

    with st.status("Running pipeline…", expanded=True) as status:

        if meta["type"] == "Microarray":
            st.write("Running microarray pipeline (async GSM fetch + GPL annotation)…")
            combined_df = fetch_microarray_matrix(meta)
        else:
            st.write(
                f"Running RNA-seq pipeline for **{gse_id}** "
                f"(GEO supplementary files → CPM → log2(CPM+1))…"
            )
            combined_df = fetch_rnaseq_matrix(gse_id)

        if combined_df is None:
            status.update(label="Pipeline failed.", state="error")
            st.stop()

        status.update(label="Done!", state="complete")

    # ── Summary metrics ───────────────────────────────────────────────────────
    n_genes   = combined_df.shape[0]
    n_samples = combined_df.shape[1] - (1 if "Name" in combined_df.columns else 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Genes",   f"{n_genes:,}")
    c2.metric("Samples", n_samples)
    c3.metric("Normalisation", "log2(CPM+1)")

    # ── Table ─────────────────────────────────────────────────────────────────
    st.dataframe(
        combined_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Name": st.column_config.TextColumn("Gene", pinned=True, width="small"),
        },
    )

    # ── Download ──────────────────────────────────────────────────────────────
    tsv = combined_df.to_csv(sep="\t", index=False)
    st.download_button(
        label="⬇️ Download as .txt (tab-separated)",
        data=tsv,
        file_name=f"{gse_id}_log2CPM.txt",
        mime="text/plain",
    )