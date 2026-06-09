import pytest
import asyncio
import httpx
import requests
import pandas as pd
import io

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
        #print(gsm_list)

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
    
async def _fetch_single_gsm(
    client: httpx.AsyncClient,
    gsm_id: str
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
                return df
    except Exception:
        pass
    return None
    
async def _inhale_all_gsms(gsm_ids: list[str]) -> list[pd.DataFrame]:
    limits = httpx.Limits(max_keepalive_connections=40, max_connections=50)
    total = len(gsm_ids)

    async with httpx.AsyncClient(limits=limits, verify=False) as client:
        tasks = [
            _fetch_single_gsm(client, gsm_id)
            for i, gsm_id in enumerate(gsm_ids)
        ]
        results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]

def test_fetch_meta_data():
    meta = fetch_metadata("GSE190275")
    print(meta)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    all_dfs = loop.run_until_complete(_inhale_all_gsms(meta["gsm_ids"]))
    assert len(all_dfs) == 60 #makes sure all samples are downloaded
    assert all_dfs[0].shape[0] == 137 #makes sure all samples have all genes
    cleaned = []
    for df in all_dfs:
        df = df[~df.index.astype(str).str.contains("AFFX")]
        if "N/A" in df.index:
            df = df.drop("N/A")
        cleaned.append(df)
    final_df = pd.concat(cleaned, axis=1, join="outer")
    assert final_df.shape[1] == 60 #makes sure all samples are included in the final merged dataframe

    meta = fetch_metadata("GSE117525")
    print(meta)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    all_dfs = loop.run_until_complete(_inhale_all_gsms(meta["gsm_ids"]))
    assert len(all_dfs) == 259 #makes sure all samples are downloaded
    assert all_dfs[0].shape[0] == 19654 #makes sure all samples have all genes
    cleaned = []
    for df in all_dfs:
        df = df[~df.index.astype(str).str.contains("AFFX")]
        if "N/A" in df.index:
            df = df.drop("N/A")
        cleaned.append(df)
    final_df = pd.concat(cleaned, axis=1, join="outer")
    assert final_df.shape[1] == 259 #makes sure all samples are included in the final merged dataframe