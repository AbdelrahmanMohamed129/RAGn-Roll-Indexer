import pandas as pd
import os

#print current working directory
print(os.getcwd())

# df = pd.read_parquet('F:/COLLEGE/GP/Indexer/RAGn-Roll-Indexer/train-00000-of-00157.parquet')
# df.to_csv('train-00000-of-00157.csv')

import duckdb
import time

# select document with id 4 in the table using duckdb
startTime = time.time()

# duckdb.read_parquet("F:/COLLEGE/GP/Indexer/RAGn-Roll-Indexer/train-00000-of-00157.parquet")
eq = duckdb.sql("SELECT text FROM read_parquet('F:/COLLEGE/GP/Indexer/RAGn-Roll-Indexer/train-00000-of-00157.parquet') WHERE id = 1000")
print(eq.text)
endTime = time.time()
print("Time taken to select document with id 4 in the table using duckdb: ", endTime - startTime)

# read from multiple parquet files
# startTime = time.time()

# query = duckdb.sql("SELECT text FROM read_parquet('F:/COLLEGE/GP/Indexer/RAGn-Roll-Indexer/train-00000-of-00157.parquet') WHERE id = 200000")