#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import string
import math
from stemming.porter2 import stem
from data_prep import Doc, parse_docs, parse_query, my_df, avg_length, get_queries, get_folder_name
from A2_Q1 import my_bm25


# ### PRM Function

# In[2]:


# Function to perform initial ranking
def initial_ranking(query, coll, df):       
    # Calculate initial scores for the documents
    initial_scores = my_bm25(coll, query, df)
    # Select the top 10 documents based on the scores
    top_k_doc_ids = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)[:10]  # top 10
    # Retrieve the top documents
    top_k_docs = [coll[doc_id] for doc_id, _ in top_k_doc_ids]
    return top_k_docs


# In[3]:


# Function to calculate relevance model probabilities
def calculate_relevance_model(selected_docs, query_terms, smoothing=0.0001):
   
    # Initialize dictionary to store the relevance model probabilities for each term
    relevance_model = {}
    # Initialize dictionary to store the joint probabilities of terms with the query likelihood score
    joint_probabilities = {}  
    
    # Iterate through each selected document
    for doc in selected_docs:
        doc_total_terms = sum(doc.terms.values()) # Total number of terms in the document
        query_likelihood_score = 1.0  # Initialize the query likelihood score
        
        # Calculate the query likelihood score for the document
        for qt in query_terms:
            if qt in doc.terms:
                # Term exists in the document, apply Laplace smoothing to the term frequency
                # P(qi|D) = (f(qi, D) + smoothing) / (|D| + smoothing * |V|)
                query_likelihood_score *= (doc.terms[qt] + smoothing) / (doc_total_terms + smoothing * len(doc.terms)) 
            else:
                # Term does not exist in the document, use Laplace smoothing to avoid zero probability
                # P(qi|D) = smoothing / (|D| + smoothing * |V|)
                query_likelihood_score *= smoothing / (doc_total_terms + smoothing * len(doc.terms)) 
        
        # Update the joint probabilities for each term in the document
        for term, freq in doc.terms.items():
            if term not in joint_probabilities:
                joint_probabilities[term] = 0.0 # Initialize the joint probability for the term
            
            # Calculate the term probability with Laplace smoothing
            # P(w|D) = (f(w, D) + smoothing) / (|D| + smoothing * |V|)
            term_prob = (freq + smoothing) / (doc_total_terms + smoothing * len(doc.terms))
            
            # Joint probability P(w, q1, ..., qn) is approximated by summing over all documents:
            # P(w, q1, ..., qn) ≈ ΣD∈C P(D) * P(w|D) * Πi P(qi|D)
            # Note: The prior probability P(D) is usually assumed to be uniform and can be ignored.
            # The expression Πi P(qi|D) is the query likelihood score for the document D.
            joint_probabilities[term] += term_prob * query_likelihood_score  

    # Calculate the normalization constant for the joint probabilities
    # P(q1, ..., qn) ≈ Σw∈V P(w, q1, ..., qn)
    normalization_constant = sum(joint_probabilities.values())

    # Get the relevance model probabilities
    for term in joint_probabilities:
        # P(w|R) ≈ P(w, q1, ..., qn) / P(q1, ..., qn)
        relevance_model[term] = joint_probabilities[term] / normalization_constant

    return relevance_model


# In[4]:


# Function to calculate KL-divergence score for each document with Laplace smoothing
def kl_divergence_ranking(relevance_model, coll, smoothing=0.0001):
    kl_scores = {}
    
    for docID, doc in coll.items():
        kl_score = 0
        # Total number of terms in the document with smoothing
        # doc_length = |D| + smoothing * |V|
        doc_length = doc.doc_len + smoothing * len(relevance_model)
        
        for term, p_w_R in relevance_model.items():
            # Frequency of the term in the document with smoothing
            # term_frequency = f(w, D) + smoothing
            term_frequency = doc.terms.get(term, 0) + smoothing
            # P(w|D) = (term frequency in document + smoothing) / (document length + smoothing * vocabulary size)
            # P(w|D) = (f(w, D) + smoothing) / (|D| + smoothing * |V|)
            p_w_D = term_frequency / doc_length
            
            if p_w_D > 0:  # Avoid log(0)
                # KL-divergence component: P(w|R) * log(P(w|D))
                kl_score += p_w_R * math.log(p_w_D)
        
        # Store the KL-divergence score for this document
        kl_scores[docID] = kl_score
    
    return kl_scores

# Explanation of the key components:
# p_w_R is obtained directly from the relevance_model dictionary.

# P(w|D) = The probability of term w appearing in a specific document D
# P(w|D) = (f(w, D) + smoothing) / (|D| + smoothing * |V|) 
# Where:
# - f(w, D) is the frequency of term w in document D
# - |D| is the total number of terms in document D
# - |V| is the vocabulary size (number of unique terms in the relevance model)
# - smoothing is a small constant added to handle zero probabilities

# The formula ensures that every term has a non-zero probability, thus avoiding issues with zero probabilities.


# In[5]:


# Main function to execute the My_PRM algorithm
def My_PRM(queries, data_path, top_k=10):
    
    curr_path=os.getcwd()
    
    # Load stop words
    stopwords_f = open('common-english-words.txt', 'r')
    stop_words = stopwords_f.read().split(',')
    stopwords_f.close()
    
    # Create folder
    output_path=curr_path+"/RankingOutputs"
    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)
    
    # Process each query and corresponding data collection
    for query_id, query in queries.items():
        matched_folder=get_folder_name(query_id, data_path)   # Get the name of corresponding collection 
        coll = None  # Initialize coll here
        
        for folder_name in os.listdir(data_path):                   
            if (folder_name == matched_folder):               # Find corresponding collection name
                inputpath=data_path+'/'+folder_name           
                coll = parse_docs(stop_words,inputpath)  # Parse the documents in the collection
                os.chdir(curr_path)
                df=my_df(coll)                            # Get a dictionary of term and document-frequency (df)
                ndocs=len(coll)                           # Count number of documents in the collection
                file_name = f"My_PRM_{query_id}Ranking.dat"
                file_path = os.path.join(output_path, file_name)
            
        if coll is None:
            print(f"No data collection found for query {query_id}")
            continue
            
        # Step 1 & 2: Initial Ranking and select top ranked documents
        top_docs = initial_ranking(query, coll, df)
        
        # Step 3: Calculate Relevance Model P(w|R)
        query_terms = parse_query(query, stop_words)
        try:
            relevance_model = calculate_relevance_model(top_docs, query_terms)
        except ValueError as e:
            print(f"Error processing query {query_id}: {e}")
            continue
        
        # Step 4: Re-rank using KL-divergence
        kl_scores = kl_divergence_ranking(relevance_model, coll)
        ranked_docs = sorted(kl_scores.items(), key=lambda item: item[1], reverse=True)
        
        with open(file_path, "w") as writer:
            query_details = f'For query "{query}", N: {ndocs}\n'
            #print(query_details)
            #writer.write(query_details)
            
            for docID, score in ranked_docs:
                details = f"{docID} {score}\n"
                #print(details)
                writer.write(details)


# In[6]:


# Execute the My_PRM algorithm for the given inputs
curr_path=os.getcwd()
data_path = curr_path+'/Data_Collection'
queries = get_queries("the50Queries.txt")
My_PRM(queries, data_path)

