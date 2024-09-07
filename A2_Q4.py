#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import string
import math
from stemming.porter2 import stem


# In[2]:


from data_prep import Doc, parse_docs, parse_query, my_df, avg_length, get_queries, get_folder_name
from A2_Q1 import BM25
from A2_Q2 import JMLM
from A2_Q3 import My_PRM


# In[3]:


curr_path=os.getcwd()
data_path = curr_path+'/Data_Collection'
queries = get_queries("the50Queries.txt")

BM25(queries, data_path)
JMLM(queries, data_path)
My_PRM(queries, data_path)


# In[4]:


collection_path=curr_path+'/RankingOutputs'

os.chdir(collection_path)

for file_ in glob.glob("*.dat"):
    file_name = os.path.basename(file_)
    print(f"{file_name}:")
    n=0
    for line in open(file_):
        line = line.strip()
        print(line)
        n+=1
        if n==15: break


# In[ ]:




