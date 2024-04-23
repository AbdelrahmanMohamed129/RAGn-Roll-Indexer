import grequests
from lxml import html
import time
import json

# # read the top k pages from file "top_k.txt"
# with open("top_k_ids.txt", "r") as f:
#     pages = f.readlines()

# # generate urls for each page
# urls = dict()
# for page in pages:
#     actual_page = int(page)//100
#     row = int(page)%100 + 1
#     url = "https://huggingface.co/datasets/Cohere/wikipedia-22-12-simple-embeddings/viewer/default/train?p=" + str(actual_page)
#     if url not in urls:
#         urls[url] = list()
#     urls[url].append(row)

# start = time.time()


# # send requests to urls
# rs = (grequests.get(u) for u in urls)
# responses = grequests.map(rs)

# top_k_data = list()

# # parse the response and get the top k data [text]
# for response in responses:
#     content = response.text
#     parsed = html.fromstring(content)
#     # get the rows in urls[response.url]
#     rows = parsed.xpath("/html/body/div[1]/main/div/div/div/div[2]/div/table/tbody/tr")
#     for row in urls[response.url]:
#         text = rows[row-1].xpath("td[3]/div/div/text()")[0]
#         top_k_data.append(text)

# end = time.time()
# print("Time taken to crawl the top K: ",end - start) #time taken to run the code

# # write the data to file
# with open("top_k_data.txt", "w") as f:
#     for data in top_k_data:
#         f.write(data + "\n")

# import requests
# headers = {"Authorization": f"Bearer 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"}
# API_URL = "https://datasets-server.huggingface.co/rows?dataset=wiki_dpr&config=psgs_w100.multiset.exact&split=train&offset=9000000&length=1"
# def query():
#     response = requests.get(API_URL, headers=headers)
#     return response.json()
# data = query()
# # with open("nashaat.txt","w") as f:
# #     json.dump(data, f)
# print(json.dumps(data, indent=4))

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfFileSystem
token = "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"
fs = HfFileSystem(token=token)
print("authenticated")
fs.ls("datasets/wiki_dpr/blob")

# start = time.time()
# # load the dataset
# base_url = "https://huggingface.co/datasets/wiki_dpr/blob/main/data/psgs_w100/nq/train-00000-of-00157.parquet"
# data_files = {"train": base_url }
# dataset = load_dataset("parquet", data_files=data_files, split="train")
# end = time.time()
# print("Time taken to read the file: ",end - start) #time taken to run the code
# # print the row whose id is 10
# print(dataset[10])