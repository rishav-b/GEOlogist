from transformers import pipeline

_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    return _classifier

text_to_classify = """Illumina Casava1.82 software used for basecalling.
Short reads in fastQ format are processed using RAPiD, which is a RNA-Seq analysis framework developed and maintained by the Technology Development group at Icahn Institute for Genomics and MultiScale Biology. .
RAPiD uses STAR to map the short reads to the [hg19 ] reference and resultant alignment map in BAM format is quantified for gene level expression using featureCounts of the subreads package. Detailed QC metrics are generated using the RNASeQC package
Genome_build: hg19
Supplementary_files_format_and_content: tab-delimited text files include fragment per kilobase per million (FPKM) for each gene"""

label_dict = {
    "raw unnormalized counts": "raw_counts",
    "cpm normalized": "cpm",
    "RPKM or FPKM normalized": "rpkm_fpkm",
    "TPM normalized": "tpm",
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
    "log transformed expression",
    "VST or rlog variance stabilized",
    "quantile normalized",
    "TMM or RLE normalized",
    "RMA normalized microarray expression",
    "fold change between conditions",
    "z-score normalized"]

result = _get_classifier()(
    text_to_classify,
    candidate_labels=candidate_labels,
    hypothesis_template="The supplementary gene expression data contains {}.",
)
print(result)