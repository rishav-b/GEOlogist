import re
import GEOparse
import pandas as pd

REJECT_PATTERNS = [
    re.compile(r'^(NM|NR|NP|XM|XR|XP|NG|NT|NC)_\d+(\.\d+)?$'),  
    re.compile(r'^[A-Z]{1,2}\d{6,}(\.\d+)?$'),                   
    re.compile(r'^ILMN_\d+$'),                                     
    re.compile(r'^ENSG\d+|^ENST\d+|^ENSP\d+'),                  
    re.compile(r'^\d+(_[a-z]_at|_at|_x_at|_s_at)$'),             
    re.compile(r'^A_\d+_P\d+$'),                                   
    re.compile(r'^chr', re.IGNORECASE),                        
    re.compile(r'^\d+[pq]\d+'),                               
    re.compile(r'^GO:\d+$'),                                 
    re.compile(r'\.\d+$'),                                   
    re.compile(r'^\d'),
    re.compile(r'\s'),                                   
    re.compile(r'_\d{4,}'),
    re.compile(r'scl\d+.'),
    re.compile(r'RefSeq'),
    re.compile(r'\+'),
    re.compile(r'^[agctAGCT]*$'),
    re.compile(r"^.{,3}$")                                       
]

ACCEPT_PATTERNS = [
    re.compile(r'^\d{7}[A-Za-z]\d{2}[Rr][Ii][Kk]$'),  
]

def is_gene_symbol(value):
    if "//" in str(value):
        value = str(value).split("//")
        v = value[1]
        v = str(v).strip()
        return (not any(p.search(v) for p in REJECT_PATTERNS)) or any(r.search(v) for r in ACCEPT_PATTERNS)

    else:
        v = str(value).strip()
        return (not any(p.search(v) for p in REJECT_PATTERNS)) or any(r.search(v) for r in ACCEPT_PATTERNS)
    
def too_homogenous(col):
    col = list(set(col))
    return len(col) < 10

def get_gene_symbol_column(gse_id, counts_df, probe_ids=None):
    gse = GEOparse.get_GEO(geo=gse_id, destdir="./transformer/test_data", silent=True, annotate_gpl=False)

    gpl_df = None
    for gpl_name, gpl in getattr(gse, "gpls", {}).items():
        gpl_df = gpl.table
        print(gpl_name)
        print(gpl.table)

    if gpl_df is None or gpl_df.empty:
        raise ValueError(f"No platform data found for {gse}")
    
    if probe_ids is not None:
        index_set = {_normalize_id(v) for v in probe_ids}
    else:
        index_set = {_normalize_id(v) for v in counts_df.index}
    
    best_col, best_score = _find_best_id_col(index_set, gpl_df)

    if best_score < 0.01:
        print(f"  [warn] best overlap is only {best_score:.3f} — index may not match any GPL column")
    
    if gpl_df is not None:
        symbol_col = None
        symbol_col_score = 0
        for col in gpl_df.columns:
            values = gpl_df[col].replace("---", pd.NA).dropna()
            if too_homogenous(values):
                continue
            col_score = values.apply(is_gene_symbol).mean()
            #print(gpl_df[gpl_df[col].apply(lambda x: not is_gene_symbol(x))][col])
            #print(col + "\t" + str(col_score) + "\t" + str(too_homogenous(col)))
            if col_score > symbol_col_score:
                symbol_col_score = col_score
                symbol_col = col
        
        return gpl_df, best_col, symbol_col
    else:
        raise ValueError(f"No platform data found for {gse}")
    

def _strip_version(s: str) -> str:
    """ENSG00000141510.14 → ENSG00000141510, NM_001234.5 → NM_001234"""
    return re.sub(r'\.\d+$', '', s)

def _normalize_id(s: str) -> str:
    s = str(s).strip().upper()
    # float → int: "1234.0" → "1234"
    if re.match(r'^\d+\.0$', s):
        s = s[:-2]
    return s

def _overlap_score(index_set: set[str], col_vals: pd.Series) -> float:
    """Try exact match first, then version-stripped match."""
    col_set_raw = set(col_vals.apply(_normalize_id))

    # exact (normalized)
    exact = len(index_set & col_set_raw) / len(index_set)
    if exact > 0:
        return exact

    # version-stripped fallback
    index_stripped = {_strip_version(v) for v in index_set}
    col_stripped   = {_strip_version(v) for v in col_set_raw}
    return len(index_stripped & col_stripped) / len(index_set)


def _find_best_id_col(index_set: set[str], gpl_df: pd.DataFrame) -> tuple[str, float]:
    best_col   = None
    best_score = -1

    for col in gpl_df.columns:
        score = _overlap_score(index_set, gpl_df[col])
        print(f"  id candidate '{col}': {score:.3f}")
        if score > best_score:
            best_score = score
            best_col   = col

    print(f"  → best id col: '{best_col}' (score {best_score:.3f})")
    return best_col, best_score

#col = get_gene_symbol_column("GSE63596")

#print(col)