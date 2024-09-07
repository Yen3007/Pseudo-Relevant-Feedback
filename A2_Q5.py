#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import string
import math
from stemming.porter2 import stem

import pandas as pd
from statistics import mean


# In[2]:


# Get the benchmark 
def get_benchmark(input_path):
    ben = {} # A
    benFile=open(input_path, 'r')
    for line in benFile:
        line=line.strip()
        line1 = line.split()
        #print(line1)
        ben[line1[1]] = int(float(line1[2])) # Dictionary {document_id: relevance(0/1)}
    benFile.close()
    return ben


# In[3]:


# Get the ranking
def get_ranking(input_path):
    rank = {} 
    i=1
    rankOutputs=open(input_path, 'r')
    for line in rankOutputs:
        line=line.strip()
        line1 = line.split()
        #print(line1)
        #print(str(i),str(line1[0]))
        rank[str(i)] = line1[0]   #Dictionary {rank:document_id}
        i+=1
    rankOutputs.close()
    return rank


# In[4]:


# Precision
def cal_precision(rank1, ben):
    r_i=0
    map1=0.0
    rel=""
    for (n,doc_id) in sorted(rank1.items(), key=lambda x: int(x[0])): # Sort by rank
        if ben[doc_id]==1:       # If the document is relevant
            r_i +=1              # Increment the number of relevant documents
            p_i = r_i/int(n)     # Precision at rank i
            map1+=p_i            # Add the precision at rank i to the Mean Average Precision
            rel="/"
        else:                    # If the document is not relevant
            r_i = r_i            # Do nothing
            p_i = r_i/int(n)     # Precision at rank i
            rel="x"

        #print("Precision at rank ", str(n), " = ", str(r_i),"/",str(n), " = ",str(p_i),rel)
        if(int(n)==10):
            p_10=p_i
            #print("Precision at 10: ", str(p_10))
    map1 = map1/int(r_i) # Mean Average Precision
    return p_10, map1


# In[5]:


# Discounted cumulative gain
def cal_dcg(rank1, ben):
    dcg=0.0                                             # DCG
    N=len(rank1)                                        # Number of documents
    for (rank,doc_id) in sorted(rank1.items(), key=lambda x: int(x[0])): # Sort by rank
        if float(rank)==1:                              # If the rank is 1
            rel_1 = ben[doc_id]                         # Relevance of the document at rank 1
            dcg = rel_1                                 # DCG at rank 1
        else:
            rel_i = ben[doc_id]                         # Relevance of the document at rank i
            dcg +=float(rel_i)/math.log2(float(rank))   # DCG at rank i
            if(float(rank)==10):                        # If the rank is 10
                #print("DCG at rank 10: ",dcg) 
                break
    return dcg


# In[6]:


def evaluate_rankings():
    # Initialize the dictionaries to store the results
    avg_precision={}
    Precision_at_10={}
    DCG_10={}

    # Initialize the lists to store the results
    Topic=[]
    BM25_avg_precision=[]
    JMLM_avg_precision=[]
    My_PRM_avg_precision=[]
    BM25_Precision_at_10=[]
    JMLM_Precision_at_10=[]
    My_PRM_Precision_at_10=[]
    BM25_DCG_10=[]
    JMLM_DCG_10=[]
    My_PRM_DCG_10=[]

    curr_path=os.getcwd() # get the current path
    
    for i in range(101, 151):

        doc_id='R'+str(i)
        Topic.append(doc_id)

        # Get the benchmark
        ben_file='Dataset'+str(i)+'.txt'
        ben_path = curr_path+'/EvaluationBenchmark/'+ben_file
        ben=get_benchmark(ben_path)

        # Get the ranking files
        BM25_file='BM25_R'+str(i)+'Ranking.dat'
        JMLM_file='JMLM_R'+str(i)+'Ranking.dat'
        My_PRM_file='My_PRM_R'+str(i)+'Ranking.dat'
        files=[BM25_file,JMLM_file,My_PRM_file]
        for file_ in files:
            input_path = curr_path+'/RankingOutputs/'+file_
            rank=get_ranking(input_path)                            # Get the ranking
            p_10, map1 = cal_precision(rank, ben)                   # Calculate Precision at rank 10 and Mean Average Precision
            dcg_10=cal_dcg(rank, ben)                               # Calculate Discounted Cumulative Gain at rank 10
        
            if file_== BM25_file:
                BM25_avg_precision.append(map1)
                BM25_Precision_at_10.append(p_10)
                BM25_DCG_10.append(dcg_10)
            elif file_== JMLM_file:
                JMLM_avg_precision.append(map1)
                JMLM_Precision_at_10.append(p_10)
                JMLM_DCG_10.append(dcg_10)
            else:
                My_PRM_avg_precision.append(map1)
                My_PRM_Precision_at_10.append(p_10)
                My_PRM_DCG_10.append(dcg_10)   
  
    # Check if the lengths of the lists are the same
    assert len(Topic) == len(BM25_avg_precision) == len(JMLM_avg_precision) == len(My_PRM_avg_precision) == len(BM25_Precision_at_10) == len(JMLM_Precision_at_10) == len(My_PRM_Precision_at_10) == len(BM25_DCG_10) == len(JMLM_DCG_10) == len(My_PRM_DCG_10), "All arrays must be of the same length"
    
    # Store the results in a dictionary
    avg_precision['Topic']=Topic
    avg_precision['BM25']=BM25_avg_precision
    avg_precision['JMLM']=JMLM_avg_precision
    avg_precision['My_PRM']=My_PRM_avg_precision
    
    Precision_at_10['Topic']=Topic
    Precision_at_10['BM25']=BM25_Precision_at_10
    Precision_at_10['JMLM']=JMLM_Precision_at_10
    Precision_at_10['My_PRM']=My_PRM_Precision_at_10
    
    DCG_10['Topic']=Topic
    DCG_10['BM25']=BM25_DCG_10
    DCG_10['JMLM']=JMLM_DCG_10
    DCG_10['My_PRM']=My_PRM_DCG_10

    # Convert the dictionary to a data frame 
    avg_precision_df=pd.DataFrame(avg_precision, index=None)
    Precision_at_10_df=pd.DataFrame(Precision_at_10, index=None)
    DCG_10_df=pd.DataFrame(DCG_10, index=None)
    
    # Save the data frames to a csv file
    avg_precision_df.to_csv('avg_precision.csv',index=False)
    Precision_at_10_df.to_csv('Precision_at_10.csv',index=False)
    DCG_10_df.to_csv('DCG_10.csv',index=False)

    return avg_precision_df, Precision_at_10_df, DCG_10_df
    


# In[8]:


# Calculate the mean of the each column in the data frame
def calculate_mean(df):
    avg={}
    for col in df.columns:
        if col != 'Topic':
            avg[col]=round(df[col].mean(),3)
    return avg

avg_precision_df, Precision_at_10_df, DCG_10_df=evaluate_rankings()
# In[13]:


MAP=calculate_mean(avg_precision_df)
avg_P10=calculate_mean(Precision_at_10_df)
avg_DCG=calculate_mean(DCG_10_df)

print("Mean Average Precision",MAP)
print("Average Precision at 10",avg_P10)
print("Average DCG at rank 10", avg_DCG)
    
print("Average precision")
print(avg_precision_df)
    
print("Precision at rank 10")
print(Precision_at_10_df)
    
print("Discounted cumulative gain at rank 10")
print(DCG_10_df)


# In[ ]:




