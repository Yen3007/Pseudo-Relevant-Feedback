#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import glob, os
import string
import sys

import math
from stemming.porter2 import stem
from data_prep import Doc, parse_docs, parse_query, my_df, avg_length, get_queries, get_folder_name


# ## JM_LM

# In[8]:


def my_JMLM(coll, q, df):
    # Load stopwords from the file
    with open('common-english-words.txt', 'r') as stopwords_f: 
        stop_words = stopwords_f.read().split(',')

    JM_LM = {}  # Initializing a dictionary to store the scores for documents
    parsed_query = parse_query(q, stop_words)  # Parse the query

    # Initialize lambda
    ld = 0.4

    # Initialize document lengths and collection frequencies
    D_len = {}
    CF = {}
    coll_length = 0
    
    # For each of the documents in collection
    for docID, doc in coll.items():
        # Initialize document length with a small non-zero value (Taught in review)
        D_len[docID] = 0.5
        for term, freq in doc.terms.items():
            # Document length is updated
            D_len[docID] += freq
            # Then the term freq for the collection is updated as well
            CF[term] = CF.get(term, 0) + freq
            # Meanwhile, can also count the collection length 
            coll_length += freq

    # Ensure all query terms are in the inverted list (CF), if is not then assign a 0
    for term in parsed_query:
        if term not in CF:
            CF[term] = 0

    # Calculate JM_LM score for each document
    for docID, doc in coll.items():
        JM_LM[docID] = 1  # Initialize the score for the document, as we are doing multiplication

        for term in parsed_query:  # Each term in the parsed query
            fqi = doc.terms.get(term, 0)  # Frequency of the term in the document
            cqi = CF[term]  # Frequency of the term in the collection (CF)
            JM_LM[docID] *= ((1 - ld) * (fqi / D_len[docID])) + (ld * (cqi / coll_length))  # Update the JM_LM score

    return JM_LM


# In[9]:


def JMLM(queries, data_path):
    curr_path=os.getcwd()
    
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()
    
    # Create folder
    folder_path=curr_path+"/RankingOutputs"
    os.makedirs(folder_path, exist_ok=True)
    
    for query_id, query in queries.items():
        matched_folder=get_folder_name(query_id, data_path)   # Get the name of corresponding collection 
        for folder_name in os.listdir(data_path):                   
            if (folder_name == matched_folder):               # Find corresponding collection name
                inputpath=data_path+'/'+folder_name           
                coll = parse_docs(stop_words,inputpath)  # Parse the documents in the collection
                os.chdir(curr_path)
                df=my_df(coll)                            # Get a dictionary of term and document-frequency (df)
                ndocs=len(coll)                           # Count number of documents in the collection
                file_name="JMLM_"+query_id+"Ranking.dat"
                file_path = os.path.join(folder_path, file_name)

        with open(file_path, "w") as writer:
            JMLM=my_JMLM(coll, query, df)
            query_details = 'For query "'+query+'", N:'+str(ndocs)+"\n"
            #print(query_details)

            for docID, score in sorted(JMLM.items(),key=lambda x: x[1],reverse=True): 
                details=str(docID)+" "+str(score)+"\n"
                #print(details)
                writer.write(details)

        writer.close()


# In[10]:


curr_path=os.getcwd()
data_path = curr_path+'/Data_Collection'
queries = get_queries("the50Queries.txt")

JMLM(queries, data_path)

