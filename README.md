# Pseudo-Relevant-Feedback
This project is build under my groupmate and I. We are findings ways to improve pseudo relevant model through probabilistic model and smoothing techniques. Kindly read the user manual to run the file.

FILES SETUP
Outputs: 
- RankingOutputs: This folder stores the output files generated by the three models, including the ranking results for each query. 
- avg_precision.csv: This file contains the average precision values calculated for three models on each query. 
- DCG_10.csv: This file contains the Discounted Cumulative Gain (DCG) at  rank 10 calculated for three models on each query. 
- Precision_at_10.csv: This file contains the precision values at rank 10  calculated for three models on each query. 

Libraries Used: 
- stemming: This folder contains files used for stemming words during text processing. 
- common-english-words.txt: This file contains a list of common English  stopwords that are excluded during text processing. 

Python for Data Processing: 
- Python scripts (A2_Q1.py, A2_Q2.py, A2_Q3.py, A2_Q4.py, A2_Q5.py): These .py files are imported into Jupyter notebooks for further analysis. 
- Jupyter Notebook (A2_Q1.ipynb, A2_Q2.ipynb, A2_Q3.ipynb,  A2_Q4.ipynb, A2_Q5.ipynb, A2_Q6.ipynb): These jupyter notebook files  are used to conveniently generate outputs for the questions. 
- Python script for Data Preparation (data_prep.py): This is the python script used for data preprocessing (Parsing Documents, Queries, Average  Document Length, Inverse Document Frequency). 

HOW TO EXECUTE THE PYTHON FILES
1) Execute A2_Q4.ipynb to generate the three models’ ranking for each query into the ‘RankingOutputs’ folder. A2_Q4.ipynb imports the models function from A2_Q1.py 
(BM25), A2_Q2.py (JMLM), A2_Q3.py (Pseudo Relevant Feedback). 
2) Execute A2_Q5.ipynb to obtain the dataframes and their csv files containing models performance results. Additionally, the dataframe will be used to compute their average values and print both the results and their average scores out in the console. 
3) Execute A2_Q6.ipynb to obtain the t-test results. 

EXPECTED OUTPUT FOR EACH PYTHON FILES
- A2_Q4.ipynb: Outputs will be text file for each model and each query, stored in the  RankingOutput folder. For instance, “BM25_R101Ranking.dat” indicates the ranking  results for BM25 on query R101. 
- A2_Q5.ipynb: Outputs will be the dataframes of performance measures for all three models and their csv files, including avg_precision.csv, DCG_10.csv and 
Precision_at_10.csv. For instance, avg_precision.csv contains the average precision values calculated for three models on each query. 
- A2_Q6.ipynb: There will not be any output files, but the t-test result is reflected under the python output itself. 
