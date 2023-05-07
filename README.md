●	Reproducibility steps
1. Install all required environment and dependencies (see dependencies section)
2. Fine-tune the six domain-specific BERT models (see training code section)
3. Anlyze and choose the best performing models for the reader model and retrieval model (PubMedBERT + BioBERT BioASQ).
4. Perform intrinsic evaluation using the code in the SleepQA_Reproduction/evaluation folder.
5. Perform extrinsic evaluation using the code in the SleepQA_Reproduction/evaluation folder and SleepQA_Reproduction/evaluation/utils folder.



●	Citation to the original paper

Iva Bojic, Qi Chwen Ong, Megh Thakkar, Esha Kamran, Irving Yu Le Shua, Jaime Rei Ern Pang, Jessica Chen, Vaaruni Nayak, Shafiq Joty, Josip Car. (2022). SleepQA: A Health Coaching Dataset on Sleep for Extractive Question Answering. Proceedings of the 2nd Machine Learning for Health symposium, PMLR 193:199-217

●	Link to the original paper’s repo 

https://github.com/IvaBojic/SleepQA

●	Dependencies

        "faiss-cpu>=1.6.1",
        "filelock",
        "numpy",
        "regex",
        "torch>=1.5.0",
        "transformers>=4.3",
        "tqdm>=4.27",
        "wget",
        "spacy>=2.1.8",
        "hydra-core>=1.0.0",
        "omegaconf>=2.0.1",
        "jsonlines",
        "soundfile",
        "editdistance",
        "jnius",
        "pyserini",
        "transformers",
        "pandas",
        "nltk.tokenize",
        "f1_score",
        "argparse",
        "pathlib",
        "seaborn",
        "matplotlib",
        "iteration_utilities",
        "re",
        "sklearn",
        "torch",
        "dpr.indexer.faiss_indexers",
        "setuptools"
        
        

●	Data download instruction
The data for fine-tuning the pretrained models is provided in the data folder in this repo, which is copied from the original paper's repo.

●	Preprocessing

The data provided fro mthe original paper's repo is already proprocessed, and therefore not further preprocessing code is needed.

●	Training code + command (if applicable)

The models used in the project are pre-trained. The fine-tuning processes are done using the DPR framework provided by Meta. The fine-tuning code are located at the SleepQA_Reproduction/models/DPR-main folder.
The example command to train train different domain-specific BERT retrieval models. 

        python train_dense_encoder.py \
        train_datasets=[sleep_train] \
        dev_datasets=[sleep_dev] \
        output_dir="PubMedBERT_full/retrieval/"

The example command to train train different domain-specific BERT reader models.

        python train_extractive_reader.py \
        encoder.sequence_length=300 \
        train_files="../../../../data/training/oracle/sleep-train.json" \
        dev_files="../../../../data/training/oracle/sleep-dev.json"  \
        output_dir="PubMedBERT_full/reader/"

●	Evaluation code + command (if applicable)

The evaluation code are located at the SleepQA_Reproduction/eval folder. You can use the following command for it:

        python __main__.pu
        

●	Pretrained model (if applicable)
There are five domain-specific BERT models used in the project:
        BioBERT
        BioBERT BioASQ
        ClinicalBERT
        SciBERT and
        PubMedBERT

●	Table of results (no need to include additional experiments, but main reproducibility result should be included)

Table 
| Model Name                                    | recall@1    | EM    | F1    |
| :---:                                         | :---:       | :---: | :---: |
| Lucene BM25 (retrieval)                       | 0.61        |       |       |
| BERT SQuAD2 (reader)                          |             | 0.5   | 0.64  |
| Fine-tuned BERT (retrieval/reader)            | 0.33        | 0.55  | 0.65  |
| Fine-tuned BioBERT (retrieval/reader)         | 0.34        | 0.59  | 0.68  |
| Fine-tuned BioBERT BioASQ (reader)            |             | 0.63  | 0.74  |
| Fine-tuned ClinicalBERT (retrieval/reader)    | 0.33        | 0.58  | 0.71  |
| Fine-tuned SciBERT (retrieval/reader)         | 0.36        | 0.59  | 0.69  |
| Fine-tuned PubMedBERT (retrieval/reader)      | 0.41        | 0.61  | 0.73  |


Table: QA pipeline evaluation results (PubMedBERT + BioBERT BioASQ denoted as pipeline 1, Lucene BM25 + BERT SQuAD2  denoted as pipeline 2)
| Pipeline Name | EM    | F1    |
| :---:         | :---: | :---: |
| Pipeline 1    | 0.25  | 0.32  |
| Pipeline 2    | 0.30  | 0.41  |

