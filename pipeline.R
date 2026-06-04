#!/usr/bin/env Rscript
# pipeline.R — Called by app.py for RNA-seq datasets
# Usage: Rscript pipeline.R <SRP_ID>
# Output: TSV to stdout (Name, GSM1, GSM2, ...)

suppressPackageStartupMessages({
  library(recount3)
  library(org.Hs.eg.db)
  library(AnnotationDbi)
  library(Matrix)
  library(curl)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: Rscript pipeline.R <SRP_ID>", call. = FALSE)
srp_id <- args[1]

sink(stderr(), type = "message")

recount3_url <- getOption("recount3_url", "http://duffel.rail.bio/recount3")

# Build URLs directly using recount3's internal shard function —
# annotation_ext() is a pure lookup (no network), so this is instant.
# The shard is the last 2 chars of the accession ID (recount3 convention).
shard <- substr(srp_id, nchar(srp_id) - 1, nchar(srp_id))
ann   <- recount3:::annotation_ext("human", "gencode_v26")  # returns "G026"

url_sra_meta <- sprintf(
  "%s/human/data_sources/sra/metadata/%s/%s/sra.sra.%s.MD.gz",
  recount3_url, shard, srp_id, srp_id
)
url_counts <- sprintf(
  "%s/human/data_sources/sra/gene_sums/%s/%s/sra.gene_sums.%s.%s.gz",
  recount3_url, shard, srp_id, srp_id, ann
)

all_urls <- c(sra_meta = url_sra_meta, counts = url_counts)

# --- PARALLEL DOWNLOAD ---
dl_dir    <- file.path(tempdir(), paste0("r3_", srp_id))
dir.create(dl_dir, showWarnings = FALSE)
destfiles <- file.path(dl_dir, basename(all_urls))
names(destfiles) <- names(all_urls)

invisible(capture.output(
  curl::multi_download(unname(all_urls), unname(destfiles), resume = FALSE, progress = FALSE),
  type = "output"
))

for (nm in names(destfiles)) {
  f <- destfiles[[nm]]
  if (!file.exists(f) || file.size(f) == 0)
    stop(sprintf("Failed to download %s — study may not be in recount3: %s", nm, all_urls[[nm]]))
}

# --- SAMPLE NAME MAPPING ---
meta_raw <- read.delim(destfiles["sra_meta"], stringsAsFactors = FALSE, check.names = FALSE)
srr_ids  <- meta_raw$external_id
gsm_ids  <- meta_raw$sra_sample_name
if (is.null(gsm_ids) || all(is.na(gsm_ids))) gsm_ids <- srr_ids

# --- COUNT MATRIX ---
count_matrix <- recount3:::read_counts(destfiles["counts"], samples = srr_ids)
colnames(count_matrix) <- gsm_ids

# --- Ensembl -> Symbol: drop anything that doesn't map ---
ensembl_clean <- gsub("^(ENS[A-Z]*)([0-9]+).*", "\\1\\2", rownames(count_matrix))
valid_ensg    <- grepl("^ENS[A-Z]+[0-9]+$", ensembl_clean)

gene_symbols <- rep(NA_character_, length(ensembl_clean))
if (any(valid_ensg)) {
  gene_symbols[valid_ensg] <- tryCatch(
    mapIds(org.Hs.eg.db,
           keys      = ensembl_clean[valid_ensg],
           column    = "SYMBOL",
           keytype   = "ENSEMBL",
           multiVals = "first"),
    error = function(e) {
      message("mapIds error: ", conditionMessage(e))
      NA_character_
    }
  )
}

# Keep only rows with a real symbol — unmapped genes are dropped entirely
keep         <- !is.na(gene_symbols)
count_matrix <- count_matrix[keep, , drop = FALSE]
symbols      <- gene_symbols[keep]

# --- CPM + log2: stay sparse until the last moment ---
lib_sizes <- Matrix::colSums(count_matrix)
cpm_mat   <- Matrix::t(Matrix::t(count_matrix) / (lib_sizes / 1e6))
log_cpm   <- log2(as.matrix(cpm_mat) + 1)
rm(cpm_mat)

# --- Output ---
final <- data.frame(Name = symbols, log_cpm, check.names = FALSE, stringsAsFactors = FALSE)
write.table(final, file = stdout(), sep = "\t", quote = TRUE, row.names = FALSE, qmethod = "double")