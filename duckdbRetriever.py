import duckdb
import time
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch
import sys
import os
import faiss
import numpy as np

from multiprocessing import Process

os.environ['KMP_DUPLICATE_LIB_OK']='True'
path = "D:/Boody/GP/wikidpr/data/psgs_w100/nq/"

# TODO: 1) Remove GT From CPP  ==> DONE
#       2) Parallelize
#       3) Get documents from duckdb ==> DONE
#       4) Remove GT.txt files

class Retrieve:

    def readQuery(self, query):
        # tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        # model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

        # #save tokenizer and model to file
        # tokenizer.save_pretrained('./Data/Model')
        # model.save_pretrained('./Data/Model')

        # read tokenizer and model from file
        tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('./Data/Model')
        model = DPRQuestionEncoder.from_pretrained('./Data/Model')

        # encode the query
        input_ids = tokenizer(query, return_tensors="pt")["input_ids"]
        embeddings = model(input_ids).pooler_output
        torch.set_printoptions(precision=10)
        with open('./Data/query.txt', 'w') as f:
            f.write(str(embeddings.detach().numpy()).replace('[','').replace(']','').replace(',','').replace('\n',''))
            f.close()

        print("####### DONE EMDEDING THE QUERY #######")


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

        # close the file and index
        f.close()
        del index
        
        print("********* Getting nearest clusters done *********")


    def cppRetrieve(self):
        startTime = time.time()
        for file_no in range(40):
            # self.getNearestClusters(file_no)
            headerFile = "D:/Boody/GP/Indexer/RAGn-Roll-Indexer/";
            # Arguments are topK (int), headerFile (string)
            topK = 10
            command = f"{headerFile}cmake-build-debug/RAGn_Roll_Indexer.exe {topK} {headerFile} {file_no} -lboost_serialization -lboost_system"
            print(command)
            os.system(command)
            print(f"####### FILE {file_no} DONE #######")

        print("####### DONE GETTING THE NEAREST CLUSTERS #######")
        endTime = time.time()
        print("Time taken to retrieve documents from c++: ", endTime - startTime)


    def loadResultset(self, k = 10):
        # load all res files in a res listm each line contains id, distance
        self.res = []
        for file_no in range(40):
            with open(f"./Data/Data_{file_no}/res.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(" ")
                    self.res.append((line[0], line[1]))
        print("####### DONE READING RESULTS #######")

        # self.gt = []
        # for file_no in range(40):
        #     with open(f"./Data/Data_{file_no}/gt.txt", "r") as f:
        #         lines = f.readlines()
        #         for line in lines:
        #             line = line.split(" ")
        #             self.gt.append((line[0], line[1]))
        # print("####### DONE READING GROUNDTRUTH #######")
        # sort descending the res list by distance which is the second value
        self.res.sort(key=lambda x: x[1], reverse=True)
        # self.gt.sort(key=lambda x: x[1], reverse=True)
        self.res = self.res[:k]
        # self.gt = self.gt[:k]


    def getRecall(self):
        # get tok 10 elements from res and gt and calculate the recall
        print("Res: ", self.res)
        print("GT: ", self.gt)
        # calculate the intersection based on the first value which is the id
        intersection = set([i[0] for i in self.res]).intersection(set([i[0] for i in self.gt]))
        self.recall = len(intersection)/10
        print("Recall: ", self.recall)


    def getDocs(self):
        # select document with id 4 in the table using duckdb
        startTime = time.time()

        ids = [int(i[0]) for i in self.res]
        fileNames = set()
        firstSplit = 133856 * 65

        for id in ids:
            if id < firstSplit:
                idx = id // 133856
            else:
                idx = 65 + (id - firstSplit) // 133855

            fileNames.add(f"{path}train-{str(idx).zfill(5)}-of-00157.parquet")

        fileNames = str(list(fileNames))
        ids = str(tuple(ids))
        eq = duckdb.sql(f"SELECT text FROM read_parquet({fileNames}) WHERE id in {ids}")
        eq = eq.fetchall()
        
        # with open("./Data/retrievedDocs.txt", "w") as f:
        #     for i in eq:
        #         f.write(str(i[0].encode('utf-8')) + "\n")
        #         # f.write(str(i) + "\n")
        #     f.close()
        
        # convert the retrieved documents to a list of strings
        self.retrievedDocs = [i[0] for i in eq]

        endTime = time.time()
        print("Time taken to retrieve documents using duckdb: ", endTime - startTime)


    def run(self, query):

        # indexerRetrieve = Retrieve()
        # query = sys.argv[1]
        self.readQuery(query)
        for file_no in range(40):
            self.getNearestClusters(file_no)

        self.cppRetrieve()

        self.loadResultset()
        # indexerRetrieve.getRecall()
        self.getDocs()

        return self.retrievedDocs
