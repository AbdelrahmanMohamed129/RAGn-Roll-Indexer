import torch
import clip
from PIL import Image, PngImagePlugin
import duckdb
import time
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import os
import faiss
import numpy as np
import cohere


os.environ['COHERE_API_KEY'] = 'gtNwvMCXjn0HnBpZ42YEbMMoXz6BqDvXY2aoQRoM'
co = cohere.Client(os.environ['COHERE_API_KEY'])

os.environ['KMP_DUPLICATE_LIB_OK']='True'

Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 1024*1024*10

device = "cuda" if torch.cuda.is_available() else "cpu"

textParquetPath = "D:/Boody/GP/wikidpr/data/psgs_w100/nq/"
imageParquetPath = "D:/Boody/GP/witbaseClean/"
textPath = "Z:/Data/"
imagesPath = "Z:/DataImgs/"
# headerPath = "D:/Boody/GP/Indexer/RAGn-Roll-Indexer/"

# TODO: 1) Remove GT From CPP           ==> DONE
#       2) Parallelize                  ==> DONE
#       3) Get documents from duckdb    ==> DONE
#       4) Remove GT.txt files          ==> DONE

class Retrieve:

    def readQueryText(self, query, model, tokenizer):
        # tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        # model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")

        # #save tokenizer and model to file
        # tokenizer.save_pretrained(self.headerPath + 'Model')
        # model.save_pretrained(self.headerPath + 'Model')

        # read tokenizer and model from file
        # tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.headerPath + 'Model')
        # model = DPRQuestionEncoder.from_pretrained(self.headerPath + 'Model')

        # encode the query
        input_ids = tokenizer(query, return_tensors="pt")["input_ids"]
        embeddings = model(input_ids).pooler_output
        torch.set_printoptions(precision=10)
        with open(self.headerPath + 'query.txt', 'w') as f:
            f.write(str(embeddings.detach().numpy()).replace('[','').replace(']','').replace(',','').replace('\n',''))
            f.close()

        print("####### DONE EMDEDING THE QUERY #######")


    def readQueryImgs(self, query, model, device):
        # model = clip.load('ViT-L/14', device=device)
        model, _ =  model

        text = clip.tokenize(query).to(device)

        textEmbed = model.encode_text(text)
        textEmbed = textEmbed.detach().cpu().numpy()[0]
        with open(self.headerPath + "query.txt", 'w') as f:
            f.write(str(np.array(textEmbed).astype("float32")).replace('[','').replace(']','').replace(',','').replace('\n',''))
            f.close()

        print("####### DONE EMDEDING THE QUERY #######")


    def getNearestClusters(self, file_no):
        # read the faiss index
        index = faiss.read_index(self.headerPath + f"Data_{file_no}/KMeansFAISS/index.bin")

        with open(self.headerPath + "query.txt", "r") as f:
            lines = [np.fromstring(line, sep=" ") for line in f]
            queries = np.array(lines)

        # Normalize the query
        queries = queries / np.linalg.norm(queries)

        _, I = index.search(queries, 10)
        I_flat = [i for i in I]
        I_flat = np.array(I_flat)

        # write the labels to a file where I_flat is the centroid index for each embedding
        with open(self.headerPath + f"Data_{file_no}/labels/query_faiss_labels.txt", "w") as f:
            f.write(str(I_flat).replace("[", "").replace("]", "").replace(",", "").replace("  ", " ").strip())

        # close the file and index
        f.close()
        del index
        
        print("********* Getting nearest clusters done *********")

    # def cppRetrieveWorker(self, file_no):
    #     # self.getNearestClusters(file_no)
    #     headerFile = "D:/Boody/GP/Indexer/RAGn-Roll-Indexer/";
    #     # Arguments are topK (int), headerFile (string)
    #     topK = 10
    #     command = f"{headerFile}cmake-build-debug/RAGn_Roll_Indexer.exe {topK} {self.headerPath} {file_no} -lboost_serialization -lboost_system"
    #     print(command)
    #     os.system(command)
    #     print(f"####### FILE {file_no} DONE #######")


    # def cppRetrieve(self):
    #     startTime = time.time()

    #     with Pool(10) as p:
    #         p.map(self.cppRetrieveWorker, range(40))      

    #     print("####### DONE GETTING THE NEAREST CLUSTERS #######")
    #     endTime = time.time()
    #     print("Time taken to retrieve documents from c++: ", endTime - startTime)


    def cppRetrieve(self):
        startTime = time.time()

        # self.getNearestClusters(file_no)
        headerFile = "D:/Boody/GP/Indexer/RAGn-Roll-Indexer/"
        # headerFile = "Z:/";
        # Arguments are topK (int), headerFile (string)
        
        for file_no in range(self.fileCount):
            command = f"{headerFile}cmake-build-debug/RAGn_Roll_Indexer.exe {self.topK} {self.headerPath} {file_no} -lboost_serialization -lboost_system"
            # print(command)
            os.system(command)
            # print(f"####### FILE {file_no} DONE #######")

        print("####### DONE GETTING THE NEAREST CLUSTERS #######")
        endTime = time.time()
        print("Time taken to retrieve documents from c++: ", endTime - startTime)


    def loadResultset(self):
        # load all res files in a res listm each line contains id, distance
        self.res = []
        for file_no in range(self.fileCount):
            with open(self.headerPath + f"Data_{file_no}/res.txt", "r") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split(" ")
                    self.res.append((line[0], line[1]))
        print("####### DONE READING RESULTS #######")
        self.res.sort(key=lambda x: x[1], reverse=True)
        self.res = self.res[:self.topK]

        # self.gt = []
        # for file_no in range(self.fileCount):
        #     with open(self.headerPath + f"Data_{file_no}/gt.txt", "r") as f:
        #         lines = f.readlines()
        #         for line in lines:
        #             line = line.split(" ")
        #             self.gt.append((line[0], line[1]))
        # print("####### DONE READING GROUNDTRUTH #######")
        # self.gt.sort(key=lambda x: x[1], reverse=True)
        # self.gt = self.gt[:self.topK]


    def getRecall(self):
        # get tok 10 elements from res and gt and calculate the recall
        print("Res: ", self.res)
        print("GT: ", self.gt)
        # calculate the intersection based on the first value which is the id
        intersection = set([i[0] for i in self.res]).intersection(set([i[0] for i in self.gt]))
        self.recall = len(intersection)/10
        print("Recall: ", self.recall)


    def getDocsText(self, query):
        # select document with id 4 in the table using duckdb
        startTime = time.time()

        # ids = [int(i[0]) for i in self.res]
        ids = self.res
        
        fileNames = set()
        firstSplit = 133856 * 65

        for id in ids:
            if id < firstSplit:
                idx = id // 133856
            else:
                idx = 65 + (id - firstSplit) // 133855

            fileNames.add(f"{textParquetPath}train-{str(idx).zfill(5)}-of-00157.parquet")

        fileNames = str(list(fileNames))
        idsDuck = str(tuple(ids))
        eq = duckdb.sql(f"SELECT text FROM read_parquet({fileNames}) WHERE id in {idsDuck}")
        eq = eq.fetchall()
        
        # with open(self.headerPath + "retrievedDocs.txt", "w") as f:
        #     for i in eq:
        #         f.write(str(i[0].encode('utf-8')) + "\n")
        #         # f.write(str(i) + "\n")
        #     f.close()
        
        # convert the retrieved documents to a list of strings
        retrievedDocs = [i[0] for i in eq]
        # ReRanker
        # make dictionary of ids and data as its value
        docs = {}
        for i in range(len(ids)):
            docs[retrievedDocs[i]] = ids[i]

        
        rerank_docs = co.rerank(
            query=query, documents=list(docs.keys()), top_n=50, model="rerank-english-v2.0"
        )

        res = [(docs[doc.document["text"]], doc.relevance_score) for doc in rerank_docs]
        
        # sort the res according to the first item in the tuple
        print(res)
        res.sort(key=lambda x: x[0]) 
        # concatenate each consecutive ids and give them a score of the maximum relevance_score among them
        docs = {v: k for k, v in docs.items()}
        new_res = []
        for i in range(len(res)):
            if i == 0 or res[i][0] != res[i-1][0] + 1:
                new_res.append((res[i][0], res[i][1], docs[res[i][0]], 0))
            elif res[i][0] == res[i-1][0] + 1:
                new_res[-1] = (new_res[-1][0], max(new_res[-1][1], res[i][1]), new_res[-1][2] + docs[res[i][0]], 1)

        # sort the new_res according to the relevance_score
        new_res.sort(key=lambda x: x[1], reverse=True)
        # get the top 10 documents
        
        self.retrievedDocs = [doc[2] for doc in new_res[:5]]
        
        endTime = time.time()
        print("Time taken to retrieve documents using duckdb: ", endTime - startTime)


    def getDocsImgs(self):
        # select document with id 4 in the table using duckdb
        startTime = time.time()

        ids = [int(i[0]) for i in self.res]
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
        self.retrievedImgs = [i[0] for i in eq]

        endTime = time.time()
        print("Time taken to retrieve documents using duckdb: ", endTime - startTime)


    def run(self, query, isImage, modelText = None, tokenizerText = None, modelImg = None, device = None):

        # indexerRetrieve = Retrieve()
        # query = sys.argv[1]

        self.headerPath = imagesPath if isImage else textPath
        self.isImage = isImage
        self.topK = 10
    
        if self.isImage:
            self.readQueryImgs(query, modelImg, device)
        else:
            self.readQueryText(query, modelText, tokenizerText)

        self.fileCount = len([name for name in os.listdir(self.headerPath) if name.startswith("Data")]) 

        for file_no in range(self.fileCount):
            self.getNearestClusters(file_no)

        self.cppRetrieve()
        self.loadResultset()
        # self.getRecall()

        if self.isImage:
            self.getDocsImgs()
        else:
            self.getDocsText()

        if self.isImage:
            return self.retrievedImgs
        else:
            return self.retrievedDocs
        
    def returnIds(self, query, isImage, modelText = None, tokenizerText = None, modelImg = None, device = None):

        # indexerRetrieve = Retrieve()
        # query = sys.argv[1]

        self.headerPath = imagesPath if isImage else textPath
        self.isImage = isImage
        self.topK = 50
    
        if self.isImage:
            self.readQueryImgs(query, modelImg, device)
        else:
            self.readQueryText(query, modelText, tokenizerText)

        self.fileCount = len([name for name in os.listdir(self.headerPath) if name.startswith("Data")]) 

        for file_no in range(self.fileCount):
            self.getNearestClusters(file_no)

        self.cppRetrieve()
        self.loadResultset()
        
        return [int(i[0]) for i in self.res]
    
    def returnActualData(self, ids, query):
        
        self.res = ids
        self.getDocsText(query)
        
        return self.retrievedDocs


if __name__ == '__main__':
    # calculate the time 

    tokenizerText = DPRQuestionEncoderTokenizer.from_pretrained('Z:/Data/Model/')
    modelText = DPRQuestionEncoder.from_pretrained('Z:/Data/Model/')
    
    temp = Retrieve()
    query = "When was world war II?"
    startTime = time.time()
    ids = temp.returnIds(query, isImage = False, modelText= modelText, tokenizerText= tokenizerText)
    print(ids)
    endTime = time.time()
    print("ALL Time taken ", endTime - startTime)
    # data = temp.returnActualData(ids, query)
    # print(len(data))
    # # dump docs in a json file
    # with open("./reranker.json", "w") as f:
        #     json.dump(docs, f, indent=4)
    
    
