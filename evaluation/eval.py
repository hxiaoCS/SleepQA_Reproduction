import sys 
import random, json, pprint
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.tokenize import word_tokenize
sys.path.append("../utils")

import csv, time, json, os
from pyserini.search.lucene import LuceneSearcher
from transformers import pipeline, BertForQuestionAnswering, AutoTokenizer
       
from f1_score import calculate_f1
     

def berts_top1(input_folder):

    for file in os.listdir(input_folder):      
        top1_res = 0
        
        with open (input_folder + file, "r", encoding = "utf-8") as fin:
            objs = json.load(fin)
            for obj in objs:
                flag = obj['ctxs'][0]['has_answer']
                if(flag):
                    top1_res += 1
                    
        print("recall@1 for {}: {:.2f}".format(file.split("_")[0], top1_res/len(objs)))     
        
       
def read_file(file_name):
    
    q_a = []
    
    with open(file_name, "r", encoding = "utf-8") as fin:
        for line in csv.reader(fin, delimiter = "\t"):
            
            q = line[0]
            a = line[1].replace('["', "").replace('"]', "").replace('"', '') 
            
            q_a.append([q, a])

    return q_a


def lucene_topk(input_file, index_file, top_k):

    q_a = read_file(input_file)
    
    topk_res = [0] * top_k
    start_time = time.time()
    
    # initialize sparce searcher
    simple_searcher = LuceneSearcher(index_file)
    
    run_time = int(time.time() - start_time)
    print("Initialization: {} seconds.".format(run_time))
    start_time = time.time()
    
    for q, a in q_a:

        hits = simple_searcher.search(q, top_k) 
        
        for i, hit in enumerate(hits):
            doc_id = hit.docid
            context = json.loads(simple_searcher.doc(doc_id).raw())['contents']              
            
            if(a in context):
                for j in range(i, top_k):
                    topk_res[j] += 1
                break
    
    for j in range(top_k):
        topk_res[j] /= len(q_a)
        
    print("recall@1: {}".format(topk_res[0]))  
    print("recall@k: {}".format(topk_res))          
    
    run_time = int(time.time() - start_time)
    print("Answering time: {} seconds.".format(run_time))
    

def read_json(file_name):
    
    f1_go, em_go = [], []

    with open(file_name, "r", encoding = "utf-8") as fin: 
        objs = json.load(fin)
        for obj in objs:
            gold = obj['gold_answers'][0]
            answer = obj['predictions'][0]['prediction']['text']
    
            macro_f1 = calculate_f1(gold, answer)   
            f1_go.append(macro_f1)
           
            if(macro_f1 == 1):
                em_go.append(1)
            else:
                em_go.append(0)
                
    return f1_go, em_go


def berts_em(input_folder):
    
    for file in os.listdir(input_folder):      

        f1_go, em_go = read_json(input_folder + file)
               
        print("{} --> em: {:.2f}, f1: {:.2f}".format(file.split("_")[0], 
                                                     sum(em_go)/len(em_go), sum(f1_go)/len(f1_go)))

    
def oracle_squad2(oracle_json):
  
    start_time = time.time()
    
    model_name = 'deepset/bert-base-uncased-squad2'
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)    
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    
    run_time = int(time.time() - start_time)
    print("Initialization: {} seconds.".format(run_time))
    start_time = time.time()
    
    f1_go, em_go = [], []
    
    with open(oracle_json, "r", encoding = "utf-8") as fin: 
        objs = json.load(fin)
        for obj in objs:
            q = obj['question']
            answer = obj['answers'][0]
            text = obj['ctxs'][0]['text']
            
            if(answer not in text):
                print(answer)
                continue
                    
            answer_object = nlp({'question': q, 'context': text})['answer']
            
            macro_f1 = calculate_f1(answer_object, answer)   
            f1_go.append(macro_f1)
           
            if(macro_f1 == 1):
                em_go.append(1)
            else:
                em_go.append(0)
                
    print("em: {:.2f}, f1: {:.2f}".format(sum(em_go)/len(em_go), sum(f1_go)/len(f1_go)))
    
    run_time = int(time.time() - start_time)
    print("Answering time: {} seconds.".format(run_time))
  
      
def pipeline1(input_file):
    
    f1_go, em_go = read_json(input_file)
           
    print("{} --> em: {:.2f}, f1: {:.2f}".format(input_file.split("_")[0].split("/")[-1:][0], 
                                                  sum(em_go)/len(em_go), sum(f1_go)/len(f1_go)))    
 
       
def pipeline2(input_file, index_file, flag_open, output_file):
    
    q_a = read_file(input_file)
    
    start_time = time.time()
    
    simple_searcher = LuceneSearcher(index_file)

    model_name = 'deepset/bert-base-uncased-squad2'
    model = BertForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)    
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    
    run_time = int(time.time() - start_time)
    print("Initialization: {} seconds.".format(run_time))
    start_time = time.time()
    
    f1_go, em_go = [], []
    q_a_p = {}
    
    for q, a in q_a:

        hits = simple_searcher.search(q)            
        doc_id = hits[0].docid
           
        context = json.loads(simple_searcher.doc(doc_id).raw())['contents']

        answer_object = nlp({'question': q, 'context': context})['answer']

        if (not flag_open):
            macro_f1 = calculate_f1(answer_object, a)   
            f1_go.append(macro_f1)
        
            if(macro_f1 == 1):
                em_go.append(1)
            else:
                em_go.append(0)
        else:
            q_a_p[q] = (context, answer_object)
    
    if (not flag_open):
        print("pipeline2 --> em: {:.2f}, f1: {:.2f}".format(sum(em_go)/len(em_go), sum(f1_go)/len(f1_go)))
    else:
        with open(output_file, "w", encoding = "utf-8") as fout: 
            for q in q_a_p.keys():
                fout.write("{}\t{}\t{}\n". format(q, q_a_p[q][0], q_a_p[q][1]))

      
    run_time = int(time.time() - start_time)
    print("Answering time: {} seconds.".format(run_time))       

    def json_csv(input_file, output_file):
    
    with open(output_file, "w", encoding = "utf-8") as fout:
        with open(input_file, "r", encoding = "utf-8") as fin: 
            objs = json.load(fin)
            for obj in objs:
                q = obj['question']
                a = obj['predictions'][0]['prediction']['text']
                p = obj['predictions'][0]['prediction']['passage']
                
                fout.write("{}\t{}\t{}\n". format(q, p, a))


def randomize_answers(pipeline1, pipeline2, output_file):
    
    pipeline1 = pd.read_csv(pipeline1, delimiter = "\t", 
                              names = ['q_p1', 'p_p1', 'a_p1'], header = None, encoding = 'utf-8')
    
    pipeline2 = pd.read_csv(pipeline2, delimiter = "\t", 
                                  names = ['q_p2', 'p_p2', 'a_p2'], header = None, encoding = 'utf-8')
     
    df_merged = pd.merge(pipeline1, pipeline2, left_index = True, right_index = True)
       
    for qn in df_merged.index:
        row = df_merged.iloc[qn]   
        
        if(row['q_p1'] != row['q_p2']):
            print("Questions are not the same!")
            
        choice = random.choice([True, False])
        
        if(choice == True):
            df_merged.loc[qn, 'answer_1'] = row['a_p1']
            df_merged.loc[qn, 'answer_2'] = row['a_p2']
            df_merged.loc[qn, 'par_1'] = row['p_p1']
            df_merged.loc[qn, 'par_2'] = row['p_p2']
        else:
            df_merged.loc[qn, 'answer_1'] = row['a_p2']
            df_merged.loc[qn, 'answer_2'] = row['a_p1']   
            df_merged.loc[qn, 'par_1'] = row['p_p2']
            df_merged.loc[qn, 'par_2'] = row['p_p1']
            
            
    df_merged = df_merged[['q_p1', 'a_p1', 'a_p2', 'par_1', 'par_2', 'answer_1', 'answer_2']]
    df_merged.to_csv(output_file, index = False)
    
    
def untangle_answers(input_file, output_file):
    
    scores_span, scores_par = defaultdict(int), defaultdict(int)

    # q_p1,a_p1,a_p2,par_1,par_2,answer_1,answer_2,score_a,score_par
    filled_file = pd.read_csv(input_file, encoding = 'utf-8')
     
    fact_q = ["who", "what", "where", "when", "why", "how"] 
    
    for qn in filled_file.index:
        row = filled_file.iloc[qn]  
        
        if(row['a_p1'] != row['answer_1']):
            if(row['score_a'] == 1):
                filled_file.loc[qn, 'score_answer'] = 2
            elif(row['score_a'] == 2):
                 filled_file.loc[qn, 'score_answer'] = 1
            else:
                filled_file.loc[qn, 'score_answer'] = row['score_a']
                
            if(row['score_par'] == 1):
                filled_file.loc[qn, 'score_paragraph'] = 2
            elif(row['score_par'] == 2):
                 filled_file.loc[qn, 'score_paragraph'] = 1
            else:
                filled_file.loc[qn, 'score_paragraph'] = row['score_par']
        else:
            filled_file.loc[qn, 'score_answer'] = row['score_a']
            filled_file.loc[qn, 'score_paragraph'] = row['score_par']
            
        words = word_tokenize(row['q_p1'])
            
        if(words[0] not in fact_q):
            scores_span[filled_file.loc[qn, 'score_answer']] += 1
            scores_par[filled_file.loc[qn, 'score_paragraph']] += 1
                
    print("Human evaluation scores for span answers (all): \n{}".format(filled_file['score_answer'].value_counts()))
    print("Human evaluation scores for span answers + explanations (all): \n{}".format(filled_file['score_paragraph'].value_counts())) 
    
    print("Human evaluation scores for span answers (not factual):")
    print("\n".join("{}\t{}".format(k, v) for k, v in scores_span.items()))
    
    print("Human evaluation scores for span answers + explanations (not factual):")
    print("\n".join("{}\t{}".format(k, v) for k, v in scores_par.items()))

    filled_file = filled_file[['q_p1', 'par_1', 'par_2', 'a_p1', 'a_p2', 'score_answer', 'score_paragraph']]
    
    filled_file.to_csv(output_file, index = False)


def convert(int_number):
    
    bin_number = np.zeros(4, int)
    
    bin_number[int_number - 1] = 1
    
    return bin_number


def get_ac1(preds_mat):
    if preds_mat.shape[0] == 1:
        return 0
    pi_q = np.sum(preds_mat, axis = 1) / preds_mat.shape[1]
    pey = (1 / (preds_mat.shape[0] - 1)) * np.sum(pi_q * (1 - pi_q))
    pa = (np.sum(preds_mat, axis = 1) * (np.sum(preds_mat, axis = 1)-1)) / (preds_mat.shape[1] * (preds_mat.shape[1] - 1))
    pa = np.sum(pa)
    ac1 = (pa - pey) / (1 - pey)
    ac1 *= 100
    return ac1


def calculate_gwet_AC1(input_file):
    
    agreement_file = pd.read_csv(input_file, encoding = 'utf-8')
    
    agreement_spans = agreement_file[['score_a_1', 'score_a_2', 'score_a_3', 'score_a_4', 'score_a_5']]
    agreement_exp = agreement_file[['score_p_1', 'score_p_2', 'score_p_3', 'score_p_4', 'score_p_5']]
    
    results = []
    for qn in agreement_spans.index:
        row = agreement_spans.iloc[qn]  

        preds_mat = np.stack(list(map(lambda n: convert(n), row)),  axis = -1)
        
        results.append(get_ac1(preds_mat))
        
    print("Gwet's AC1 score for span answers is: {:.2f}".format(sum(results)/len(results)))
    
    results = []
    for qn in agreement_exp.index:
        row = agreement_exp.iloc[qn]  

        preds_mat = np.stack(list(map(lambda n: convert(n), row)),  axis = -1)
        
        results.append(get_ac1(preds_mat))
        
    print("Gwet's AC1 score for span answers + explanations is: {:.2f}".format(sum(results)/len(results))) 
       
    

    
    
    
    
    
    
    