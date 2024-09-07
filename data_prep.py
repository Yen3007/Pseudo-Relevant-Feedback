#!/usr/bin/env python
# coding: utf-8

# In[2]:


import glob, os
import string
import sys

import math
from stemming.porter2 import stem


# In[3]:


class Doc:
    def __init__(self, docID, terms, doc_len):   # Initialize Doc object with 
        self.docID=docID                         # docID
        self.terms=terms                         # dictionary of terms and terms' frequency
        self.doc_len=doc_len                     # number of words in the document


# In[4]:


def parse_docs(stop_words, inputpath):
    doc_collection = {}                                         # Initialize an empty dictionary to store documents' ID and Doc objects
    os.chdir(inputpath)
    for file_ in glob.glob("*.xml"):                            # For each file in the input folder
        start_end = False                                       # Initialize a boolean variable to indicate the start and end of the document
        word_count = 0                                          # Initialize a variable to count the number of words in the document
        curr_doc = {}                                           # Initialize an empty dictionary to store terms and terms' frequency for each file                                             
        for line in open(file_):                                # For each line in the file
            line = line.strip()                                 # Remove leading and trailing whitespaces
            if(start_end == False):                             # If the start of the document is not found
                if line.startswith("<newsitem "):               # If the line starts with "<newsitem "
                    for part in line.split():                   # For each part in the line
                        if part.startswith("itemid="):          # If the part starts with "itemid="
                            docid = part.split("=")[1].split("\"")[1]   # Extract the document ID
                            break                               # Break the loop
                if line.startswith("<text>"):                   # If the line starts with "<text>"
                    start_end = True                            # Set the start of the document to True
            elif line.startswith("</text>"):                    # If the line starts with "</text>"
                break                                           # Break the loop
            else:                                               
                line = line.replace("<p>", "").replace("</p>", "")  # Remove "<p>" and "</p>" from the line
                line = line.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) # Remove digits and punctuation from the line
                for term in line.split():                        # For each term in the line
                    word_count += 1                              # words = any sequence of alphanumeric characters, terminated by a space or special character
                    term = stem(term.lower())                    # terms = words that are reduced morphological variations by stemming and converting to lower case.
                    if len(term) > 2 and term not in stop_words: # terms also have length of 3 or more and not in stop words list.
                        try:
                            curr_doc[term] += 1                  # If the term present in dictionary, increase its frequency
                        except KeyError:                         # If the term is not present in dictionary
                            curr_doc[term] = 1                   # add the term to the dictionary, and set its frequency to 1
        new_doc=Doc(docid, curr_doc, word_count)                 # Create new Doc object that has docID, dictionary of  terms and terms' frequency 
                                                                 #and number of words in the document as its attributes
        doc_collection[docid]=new_doc                            # Save the Doc object in the dictionary and use docID as dictionary key
    return doc_collection


# In[5]:


def parse_query(query0, stop_words):
    curr_doc={}     # Initialize an empty dictionary to store terms and terms' frequency in a file  
    
    # Tokenizing steps for queries is identical to steps for documents
    query0 = query0.translate(str.maketrans('','', string.digits)).translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))) # Remove digits and punctuation from the line
    for term in query0.split():                               # For each term in the query  
        term = stem(term.lower())                             # terms = words that are reduced morphological variations by stemming and converting to lower case.
        if len(term) > 2 and term not in stop_words:          # terms also have length of 3 or more and not in stop words list.
            try:                                             
                curr_doc[term] += 1                           # If the term present in dictionary, increase its frequency
            except KeyError:                                  # If the term is not present in dictionary
                curr_doc[term] = 1                            # add the term to the dictionary, and set its frequency to 1
    return curr_doc                                           # Return the dictionary of terms and terms' frequency in the query


# In[6]:


def my_df(coll):
    df_ = {}                        # Initialize an empty dictionary to store terms and terms' frequency  in the collection
    for docID, doc in coll.items():      # For each document in the collection
        for term in doc.terms.keys():    # For each term in the document
            try:
                df_[term] += 1           # If the term present in dictionary, increase its frequency
            except KeyError:             # If the term is not present in dictionary
                df_[term] = 1            # add the term to the dictionary, and set its frequency to 1
    return df_                           # Return the dictionary of terms and terms' frequency in the collection


# In[7]:


def avg_length(coll): 
    totalDocLength=0;                          # Initialize a variable to store total number of words in the collection
    N=len(coll)                                # Count number of documents in the collection
    for docID, doc in coll.items():            # For each document in the collection
        totalDocLength+=doc.doc_len            # Accumulate the number of words in the document to the total number of words in the collection
    avg_doc_length=totalDocLength/N            # Compute average document length
    return avg_doc_length                      # Return average document length


# In[8]:


# Only title of the query is used to retrieve documents
def get_queries(inputpath):                    # input path is the path of the query file
    queries = {}                               # Initialize an empty dictionary to store query_id and query
    for line in open(inputpath):               # For each line in the query file
        line = line.strip()                    # Remove leading and trailing whitespaces
        if line.startswith("<num>"):           # If the line starts with "<num>"
            query_id=line.split(": ")[1]       # Extract the query ID
        elif line.startswith("<title>"):       # If the line starts with "<title>"
            query = line.replace("<title>","").strip()  # Extract the query
            queries[query_id] = query                   # Save the query in the dictionary and use query_id as dictionary key
    return queries                             # Return the dictionary of query_id and query

# In[10]:


def get_long_queries(inputpath):
    queries = {}
    collect = False
    for line in open(inputpath):
        line = line.strip()
        if line.startswith("<num>"):
            query_id=line.split(": ")[1]
        elif line.startswith("<title>"):
            query = line.replace("<title>", "").strip()
        elif line.startswith("<desc>"):
            collect = True
        elif line.startswith("<narr>"):
            collect = True
        elif line.startswith("</Query>"):
            collect = False
        elif collect:
            query += " " + line.strip() 
            queries[query_id] = query.strip()
    return queries                         # return dictionary of query_id,query 


# In[9]:


def get_folder_name(query_id, file_path):
    query_num = query_id[-3:]                    # Get number from query_id (last 3 letters)
    for folder_name in os.listdir(file_path):    # For each folder in the file path
        if folder_name == "Data_C"+query_num:    # Find corresponding folder for the query
            return folder_name                   # Return the folder name

