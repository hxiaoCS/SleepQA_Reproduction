## eval.py

In this script, we calculate:
1. recall@k for domain-specific retrieval BERTs and Lucene BM25
2. em/f1 scores for domain-specific reader BERTs and BERT SQuAD2
3. em/f1 scores for two different Q/A pipelines
4. we prepare files for human evaluation by randomizing and untangling answers from two pipelines.

## indexes.py

This script is used to generate sparse indexes for Lucene BM25 retrieval.

## inter_agreement.py

In this script, we calculate inter-annotator agreement for label collection process and for extrinsic evaluation.
