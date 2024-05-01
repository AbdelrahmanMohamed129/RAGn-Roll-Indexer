import os
import faiss
import numpy as np
import pandas as pd
import shutil
np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # ensure numpy array won't truncate

class BuildIndex:

    def makeDirectory(self, path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of the directory %s failed" % path)
        else:
            print("Successfully created the directory %s" % path)
            

    def makeDirectories(self, file_no):
        self.makeDirectory(f"./Data/Data_{file_no}")
        self.makeDirectory(f"./Data/Data_{file_no}/clusters")
        self.makeDirectory(f"./Data/Data_{file_no}/dataset")
        self.makeDirectory(f"./Data/Data_{file_no}/indexed")
        self.makeDirectory(f"./Data/Data_{file_no}/indexed/test")
        self.makeDirectory(f"./Data/Data_{file_no}/KMeansFAISS")
        self.makeDirectory(f"./Data/Data_{file_no}/labels")
        self.makeDirectory(f"./Data/Data_{file_no}/normalizedDataset")
        

    def copyFiles(self, file_no, idx, size):
        for i in range(size):
            shutil.copy(f"D:/Boody/GP/wikidpr/data/psgs_w100/nq/train-{str(i+idx).zfill(5)}-of-00157.parquet", f"D:/Boody/GP/Indexer/RAGn-Roll-Indexer/Data/Data_{file_no}/dataset")
            print(f"Copying train-{str(i+idx).zfill(5)}-of-00157 done")
        print("********* Copying done *********")


    def readDataset(self, file_no):   
        path = f"./Data/Data_{file_no}/dataset/"

        parq = pd.read_parquet(path, engine="fastparquet", columns=["id", "embeddings"])
        # print(embeds["embeddings"][0])
        # normalize the embeddings
        parq['embeddings'] = parq['embeddings'].apply(lambda x: np.array(x) / np.linalg.norm(x) if x is not None else None)
        # print(embeds["embeddings"][0])
        self.embeds = parq['embeddings'].values
        print("Read embeddings ", self.embeds.shape)
        self.ids = parq['id'].values
        print("Read ids ", self.ids.shape)
        del parq
        self.embeds = np.vstack(self.embeds)
        print("********* Read dataset ", self.embeds.shape, "*********")


    def writeNormalizedDataset(self, file_no):
        c = 0
        for i in range(0, len(self.embeds), 100000):
            with open(f"./Data/Data_{file_no}/normalizedDataset/{c}-embeds-batch.txt", "w") as f:
                for j in range(i, min(i + 100000, len(self.embeds))):
                    f.write(str(self.ids[j]) + " ")
                    f.write(str(self.embeds[j]).replace('  ',' ').replace('[','').replace(']',''))
                    f.write('\n')
            print(f"Written {c}-embeds-batch.txt")
            c += 1
        # freeing some memory
        del self.ids
        print ("********* Writing normalized dataset done *********")


    def FAISSClustering(self, file_no):
        ncentroids = 100
        niter = 300
        verbose = True
        gpu = True
        dim = 768
        kmeans = faiss.Kmeans(dim, ncentroids, niter = niter, verbose = verbose, 
                            min_points_per_centroid = 100, max_points_per_centroid = 100000, 
                            nredo = 2, spherical = True)

        kmeans.train(self.embeds)

        faiss.write_index(kmeans.index, f"./Data/Data_{file_no}/KMeansFAISS/index.bin")
        print("********* FAISS Clustering done *********")


    def pointsAssignmentToClusters(self, file_no):
        # read the faiss index
        index = faiss.read_index(f"./Data/Data_{file_no}/KMeansFAISS/index.bin")

        # mapping each point to its nearest centroid
        _, I = index.search(self.embeds, 2)

        # flatten the array of arrays
        I_flat = [item for sublist in I for item in sublist]

        # write the labels to a file where I_flat is the centroid index for each embedding
        with open(f"./Data/Data_{file_no}/labels/faiss_labels.txt", "a") as f:
            # remove the string array representation of the list
            f.write(str(I_flat).replace("[", "").replace("]", "").replace(",", "").replace("  ", " ").strip())
        
        print("********* Points assignment to clusters done *********")


    def getNearestClusters(self, file_no):
        # read the faiss index
        index = faiss.read_index(f"./Data/Data_{file_no}/KMeansFAISS/index.bin")

        with open("./Data/query.txt", "r") as f:
            lines = [np.fromstring(line, sep=" ") for line in f]
            queries = np.array(lines)

        # Normalize the query
        queries = queries / np.linalg.norm(queries)

        _, I = index.search(queries, 10)
        I_flat = [i for i in I]
        I_flat = np.array(I_flat)

        # write the labels to a file where I_flat is the centroid index for each embedding
        with open(f"./Data/Data_{file_no}/labels/query_faiss_labels.txt", "w") as f:
            f.write(str(I_flat).replace("[", "").replace("]", "").replace(",", "").replace("  ", " ").strip())
        
        print("********* Getting nearest clusters done *********")

    # delete all the files in clusters, dataset, normalizedDataset directories
    def deleteFiles(self, file_no):
        shutil.rmtree(f"./Data/Data_{file_no}/clusters")
        shutil.rmtree(f"./Data/Data_{file_no}/dataset")
        shutil.rmtree(f"./Data/Data_{file_no}/normalizedDataset")
        self.makeDirectory(f"./Data/Data_{file_no}/clusters")
        self.makeDirectory(f"./Data/Data_{file_no}/dataset")
        self.makeDirectory(f"./Data/Data_{file_no}/normalizedDataset")


# for file_no in range(33, 40):
file_no = 39
indexer = BuildIndex()

indexer.makeDirectories(file_no)
indexer.copyFiles(file_no, file_no * 4, 4)
indexer.readDataset(file_no)
indexer.writeNormalizedDataset(file_no)
indexer.FAISSClustering(file_no)
indexer.pointsAssignmentToClusters(file_no)
indexer.getNearestClusters(file_no)

# freeing some memory
del indexer.embeds

# C++ part to build the index
headerFile = "D:/Boody/GP/Indexer/RAGn-Roll-Indexer/";
# Arguments are R(int), L(int), alpha(double), topK (int), buildIndex (bool), headerFile (string), file_no (int)
R = 20
L = 30
alpha = 1.0
buildIndex = 1
topK = 10
command = f"{headerFile}cmake-build-debug/RAGn_Roll_Indexer.exe {R} {L} {alpha} {topK} {buildIndex} {headerFile} {file_no} -lboost_serialization -lboost_system"
print(command)
os.system(command)

indexer.deleteFiles(file_no)

