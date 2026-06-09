from src.geo_rnaseq_normalizer import _scrape_geo_download_page
from src.geo_rnaseq_normalizer import _build_matrix_from_flat_files, _build_matrix_from_tar, FileMeta, _download_all
import tarfile
import gzip
import io
import pandas as pd
import numpy as np
import pytest

#makes sure the function is downloading all supplementary files AND ncbi generated files
#ensures it can categorize them correctly

def test_scrape_geo_download_page():
    assert _scrape_geo_download_page("GSE83687")[1].url == "https://www.ncbi.nlm.nih.gov/geo/download/?type=rnaseq_counts&acc=GSE83687&format=file&file=GSE83687_raw_counts_GRCh38.p13_NCBI.tsv.gz"
    assert len(_scrape_geo_download_page("GSE83687")) == 5
    assert _scrape_geo_download_page("GSE181056")[0].ncbi_data == False

def test_build_matrix_from_flat_files():
    url_bytes = _download_all(["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE181nnn/GSE181056/suppl/GSE181056_allSamples_RNASeq_RSEM.tsv.gz"])
    file_meta = [FileMeta(url      = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE181nnn/GSE181056/suppl/GSE181056_allSamples_RNASeq_RSEM.tsv.gz",
    filename = "GSE181056_allSamples_RNASeq_RSEM.tsv.gz",
    is_tar   = False,
    ncbi_data = False)]
    df = _build_matrix_from_flat_files(url_bytes, file_meta)
    assert df.shape[0] == 24454
    assert df.shape[1] == 24
    assert float(df.at['0610007P14Rik', 'UT_R2.expected_count']) == 1219.0
    #print(df)

    url_bytes = _download_all(["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE86nnn/GSE86978/suppl/GSE86978_readCounts.xls.gz"])
    file_meta = [FileMeta(url      = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE86nnn/GSE86978/suppl/GSE86978_readCounts.xls.gz",
    filename = "GSE86978_readCounts.xls.gz",
    is_tar   = False,
    ncbi_data = False)]
    df = _build_matrix_from_flat_files(url_bytes, file_meta)
    assert df.shape[0] == 21696
    assert df.shape[1] == 77
    assert int(df.at["eg:9987:chr4:m","BrTr11_CL1_022513"]) == 1733

def test_build_matrix_from_tar():
    url_bytes = _download_all(["https://ftp.ncbi.nlm.nih.gov/geo/series/GSE157nnn/GSE157878/suppl/GSE157878_RAW.tar"])
    file_meta = [FileMeta(url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE157nnn/GSE157878/suppl/GSE157878_RAW.tar",
                          filename = "GSE157878_RAW.tar",
                          is_tar = True,
                          ncbi_data = False)]
    
    df = _build_matrix_from_tar(url_bytes, file_meta)
    assert df.shape[1] == 72 #checks if it is downloading all the samples
    assert df.shape[0] == 49567 #inspected excel file and discarded metadata rows to get this figure
    assert int(df.at["0610007P14Rik","GSM4777796_1_18.counts"]) == 2868
    



