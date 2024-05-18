import pandas as pd
import time 
import duckdb
# df = pd.read_parquet(f"Z:/DataImgs/Data_2/dataset/train-{str(74).zfill(5)}-of-00330.parquet", engine='fastparquet')

# # # remove the column _index at
# # df = df.drop(columns=["__index_level_0__"])

# # save the parquet file
# # df.to_parquet(f"Z:/DataImgs/Data_2/train-{str(70).zfill(5)}-of-00330.parquet")
# df.to_parquet(f"Z:/DataImgs/Data_2/dataset/train-{str(74).zfill(5)}-of-00330.parquet",  engine='pyarrow', index=False)


# for i in range(330):
#     df = pd.read_parquet(f"D:/Boody/GP/witbaseClean/train-{str(i).zfill(5)}-of-00330.parquet", engine='pyarrow')
#     # get the count of rows in the df
#     with open("counter.txt","a") as f:
#         f.write(str(df.shape[0]))
#         f.write("\n")
imageParquetPath = "D:/Boody/GP/witbaseClean/"
startTime = time.time()

ids = [5085679, 1]
fileNames = set()
firstSplit = 19629 * 15

for id in ids:
    print(id)
    if id < firstSplit:
        idx = id // 19629
    else:
        idx = 15 + (id - firstSplit) // 19628

    fileNames.add(f"{imageParquetPath}train-{str(idx).zfill(5)}-of-00330.parquet")

fileNames = str(list(fileNames))
ids = str(tuple(ids))
eq = duckdb.sql(f"SELECT image_url FROM read_parquet({fileNames}) WHERE id in {ids}")
eq = eq.fetchall()

# with open(self.headerPath + "retrievedDocs.txt", "w") as f:
#     for i in eq:
#         f.write(str(i[0].encode('utf-8')) + "\n")
#         # f.write(str(i) + "\n")
#     f.close()

# convert the retrieved documents to a list of strings
retrievedDocs = [i[0] for i in eq]
print(retrievedDocs)

endTime = time.time()
print("Time taken to retrieve documents using duckdb: ", endTime - startTime)