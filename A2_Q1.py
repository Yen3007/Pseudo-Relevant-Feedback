#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob, os
import string
import sys

import math
from stemming.porter2 import stem
from data_prep import Doc, parse_docs, parse_query, my_df, avg_length, get_queries, get_folder_name


# In[2]:


# data_prep.py
# Doc object -> docID, terms, doc_len
# parse_docs(stop_words, inputpath)
# parse_query(query0, stop_words)
# my_df(coll)
# avg_length(coll)
# get_queries(inputpath)
# get_folder_name(query_id, file_path)


# In[3]:


# ### BM25 Function ###
### The meaning of variables from lecture slides and some notes related to written function ###
# N =  the total number of documents in the collection = len(coll)
# n_i = the number of documents that contain term i = df.value of term i
# f_i = the frequency weight (count) of term i in the document = f of doc in coll
# qf_i = the frequency weight (count) of term i in the query = f of term in q
# dl = doc length = doc.doc_len
# avdl = the average length of a document in the collection.
# k_1 = Typical TREC value for k1 = 1.2
# k_2 = 500
# b = 0.75
# K = k_1*((1-b)+(b*dl/avdl))
# R = the number of relevant documents for this query
# ri = the number of relevant documents containing term i
# The base of log is 10!!!!
# BM25 = sum of(log of (((r_i+0.5)/(R-r_i+0.5))/((n_i-r_i+0.5)/(N-n_i-R+r_i+0.5))) * ((k_1+1)*f_i)/(K+f_i) * ((k_2+1)*qf_i)/(k_2+qf_i))

# coll = dictionary of docID and Rcv1Doc object
# q =query
# df = dictionary of term and document-frequency (df)
def my_bm25(coll, q, df):
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()
    
    parsed_query=parse_query(q, stop_words) # Parse the query.
    N=len(coll)                                               # Count number of documents in the collection
    avdl=avg_length(coll)                                     # Count  the average length of a document in the collection.
    k_1 = 1.2                                                             
    k_2 = 500                                                           
    b = 0.75
    R=0
    r_i=0

    BM25={}                                              # Initialize an emtry dictionary to store docID and BM25 score of the document
    for docID, doc in coll.items():                      # For each document in the collection
        score=0                                          # Set BM25 score of each doc = 0
        for term,qf_i in parsed_query.items():           # For each document in the parsed query
            n_i=0
            if term in df.keys():
                n_i=df[term]                             #  The number of documents that contain term i
                dl=doc.doc_len                               # doc length
                K=k_1*((1-b)+(b*dl/float(avdl)))
                if term in doc.terms.keys():                 # For each document in the document
                    f_i=doc.terms[term]                      # Get term frequency  in the document
                else: f_i=0
                #term_1=math.log10(((r_i+0.5)/(R-r_i+0.5))/((n_i-r_i+0.5)/(N-n_i-R+r_i+0.5)))
                #term_1=math.log10(((N-n_i+0.5)/n_i+0.5))     # if assume R=r_i=0
                term_1=math.log10((N-n_i+0.5)/(n_i+0.5)+1)  ## check again         
                term_2=((k_1+1)*f_i)/(K+f_i)
                term_3=((k_2+1)*qf_i)/(k_2+qf_i)
                score+=(term_1*term_2*term_3)                # Accumulate score of each term
        BM25[docID]=score                                # Store docID and its BM25 score in the dictionary
    return(BM25)


# In[4]:


def BM25(queries, data_path):
    
    curr_path=os.getcwd()
    
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()
    
    # Cretae folder
    folder_path=curr_path+"/RankingOutputs"
    os.makedirs(folder_path, exist_ok=True)

    for query_id, query in queries.items():
        matched_folder=get_folder_name(query_id, data_path)   # Get the name of corresponding collection 
        for folder_name in os.listdir(data_path):                   
            if (folder_name == matched_folder):               # Find corresponding collection name
                inputpath=data_path+'/'+folder_name           
                coll = parse_docs(stop_words,inputpath)  # Get a dictionary of docID and Rcv1Doc object
                os.chdir(curr_path)
                df=my_df(coll)                            # Get a dictionary of term and document-frequency (df)
                ndocs=len(coll)                           # Count number of documents in the collection
                file_name="BM25_"+query_id+"Ranking.dat"
                file_path = os.path.join(folder_path, file_name)
        
        with open(file_path, "w") as writer:
                bm25=my_bm25(coll, query, df)
                query_details = str(query_id)+' Query:  "'+query+'", N:'+str(ndocs)+"\n"
                #print(query_details)
                
                n=0
                for docID, score in sorted(bm25.items(),key=lambda x: x[1],reverse=True): 
                    details=str(docID)+" "+str(score)+"\n"
                    n +=1 
                    #print(details)
                    writer.write(details)
                    #if n==15: break

        writer.close()
  


# In[5]:


curr_path=os.getcwd()
data_path = curr_path+'/Data_Collection'
queries = get_queries("the50Queries.txt")

BM25(queries, data_path)

