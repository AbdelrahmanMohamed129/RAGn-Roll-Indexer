# import cohere
import os
# import faiss 
# import numpy as np

# os.add_dll_directory("C://Users/BoodyBeeh/anaconda3/DLLs")

# os.environ['COHERE_API_KEY'] = 'gtNwvMCXjn0HnBpZ42YEbMMoXz6BqDvXY2aoQRoM'
# co = cohere.Client(os.environ['COHERE_API_KEY'])

# texts = ['When was Israeli Prime Minister Yitzhak Rabin assassinated']
# response = co.embed(texts=texts, model='multilingual-22-12')
# embeddings = response.embeddings # All embeddings for the texts
# print(embeddings[0][:5]) # Let's check embeddings for the first text

# with open('./Data/query.txt', 'w') as f:
#     f.write(str(embeddings).replace('[','').replace(']','').replace(',',''))
#     f.close()

# # read queries and compute nearest cloisters
# np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # ensure numpy array won't truncate

# # read the faiss index
# index = faiss.read_index("./Data/KMeansFAISS/index.bin")

# with open("./Data/query.txt", "r") as f:
#     lines = [np.fromstring(line, sep=" ") for line in f]
#     queries = np.array(lines)

# #Normalize the query
# queries = queries / np.linalg.norm(queries)

# _, I = index.search(queries, 10)
# I_flat = [i for i in I]

# # write the labels to a file where I_flat is the centroid index for each embedding
# with open("./Data/labels/query_faiss_labels.txt", "w") as f:
#     f.write(str(I_flat).replace("[", "").replace("]", "").replace(",", "").replace("  ", " ").strip())

headerFile = "D:/Boody/GP/Indexer/RAGn-Roll-Indexer/";
# Arguments are topK (int), headerFile (string)
topK = 10
command = f"{headerFile}cmake-build-debug/RAGn_Roll_Indexer.exe {topK} {headerFile} -lboost_serialization -lboost_system"
print(command)
os.system(command)

