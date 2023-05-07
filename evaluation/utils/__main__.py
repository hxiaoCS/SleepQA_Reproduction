from dataset_analysis import avg_no_words, calculate_entailment, calculate_qa_sim
from inter_agreement import labels_agreement


if __name__ == "__main__":

    agreement_file = "../data/agreement/labels_agreement.csv"
  
    # calculate em/f1 for inter-annotators agreement on labels
    labels_agreement(agreement_file)
    
