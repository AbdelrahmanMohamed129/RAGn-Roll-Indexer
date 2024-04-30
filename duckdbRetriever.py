import duckdb
import time

# select document with id 4 in the table using duckdb
startTime = time.time()

ids = range(10_000_000, 10_000_011)
fileNames = set()
firstSplit = 133856 * 65
for id in ids:
    if id < firstSplit:
        idx = id // 133856
    else:
        idx = 65 + (id - firstSplit) // 133855

    fileNames.add(f"D:/Boody/GP/wikidpr/data/psgs_w100/nq/train-{str(idx).zfill(5)}-of-00157.parquet")
fileNames = str(list(fileNames))
ids = str(tuple(ids))
# print(fileNames)
# print(ids)
eq = duckdb.sql(f"SELECT text FROM read_parquet({fileNames}) WHERE id in {ids}")
eq = eq.fetchall()
# print(eq)

# duckdb.read_parquet("F:/COLLEGE/GP/Indexer/RAGn-Roll-Indexer/train-00000-of-00157.parquet")
# for i in range(157):
#     eq = duckdb.sql("SELECT text FROM read_parquet('D:/Boody/GP/wikidpr/data/psgs_w100/nq/train-00000-of-00157.parquet') WHERE id = 1000")
#     # eq = duckdb.sql(f"SELECT count(id) as farah FROM read_parquet('D:/Boody/GP/wikidpr/data/psgs_w100/nq/train-{str(i).zfill(5)}-of-00157.parquet')")
#     with open("duckdb_output.txt", "a") as f:
#         eq = eq.fetchall()
#         f.write(str(eq[0][0]))
#         f.write('\n')

endTime = time.time()
print("Time taken to select document with id 4 in the table using duckdb: ", endTime - startTime)

# read from multiple parquet files
# startTime = time.time()

# query = duckdb.sql("SELECT text FROM read_parquet('F:/COLLEGE/GP/Indexer/RAGn-Roll-Indexer/train-00000-of-00157.parquet') WHERE id = 200000")