import torch
import clip
from PIL import Image, PngImagePlugin, UnidentifiedImageError
import tensorflow as tf
import time
from io import BytesIO
import pandas as pd
import numpy as np
from transformers import CLIPProcessor, CLIPModel

Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 1024*1024*10

device = "cuda" if torch.cuda.is_available() else "cpu"

# model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load('ViT-L/14', device=device)

# calculate the time taken
start = time.time()

for i in range(299,330):
    # read the parquet file 
    df = pd.read_parquet(f'D:/Boody/GP/witbase/train-{str(i).zfill(5)}-of-00330.parquet')
    embeds = []
    # create new column named embedding
    df["embedding"] = None
    for index in df.index:
        try:
            image_data = BytesIO(df.at[index,'image']['bytes'])
            image = Image.open(image_data)
            image = preprocess(image).unsqueeze(0).to(device)

            with torch.no_grad():
                image_features = model.encode_image(image)
            
            # save the new value of embedding and convert it to float32
            df.at[index,"embedding"]= image_features.cpu().numpy()[0].astype('float32')

            print(f"Done row {index}")
        except UnidentifiedImageError:
            print(f"========================== Cannot identify image at row {index}. Discarding row. ==========================")
            df = df.drop(index)
        
    end = time.time()
    print(f" Finished file {i} ******** Time taken: ", end - start)  
    df = df[["image_url", "embedding", "id"]]

    # save the parquet file 
    df.to_parquet(f'D:/Boody/GP/witbaseClean/train-{str(i).zfill(5)}-of-00330.parquet')

# df = pd.read_parquet('D:/Boody/GP/witbaseClean/train-{str(0).zfill(5)}-of-00330.parquet')
# # imageEmbeds = df.iloc[7]["embedding"]

# # put this str "Puerto Rican Giant Centipede" in a tf
# query_text = "Panasonic Digital Camera"
# text = clip.tokenize(query_text).to(device)

# textEmbed = model.encode_text(text)
# textEmbed = textEmbed.detach().cpu().numpy()[0]
# with open("Z:/DataImgs/query.txt", 'w') as f:
#     f.write(str(np.array(textEmbed).astype("float32")).replace('[','').replace(']','').replace(',','').replace('\n',''))
#     f.close()
# textEmbed = np.array(textEmbed) / np.linalg.norm(textEmbed)

# # loop on the first 1000 elements in the df
# imageEmbeds = []
# for index, row in df.iterrows():
#     temp = np.array(row["embedding"]) / np.linalg.norm(row["embedding"])
#     imageEmbeds.append(temp)
#     if index == 1000:
#         break

# # get the top 10 nearest embeddings to the text embedding
# similarity = []
# for image in imageEmbeds:
#     similarity.append(np.dot(image, textEmbed))

# # get the indicies of the to 10 nearest similarities in the similarity array
# similarity = np.array(similarity)
# print(sorted(similarity)[-10:])
# print(np.argsort(similarity)[-10:])
# arr = [815  ,36 ,851 ,490 ,508, 733  ,75 ,518 ,541 ,795]
# for i in range(len(arr)):
#     print(arr[i])
#     print(df["image_url"][arr[i]])
# # imageEmbeds = np.array(imageEmbeds) / np.linalg.norm(imageEmbeds)

# # print(len(imageEmbeds))
# # print(len(textEmbed))
# # similarity = np.dot(imageEmbeds, textEmbed)
# # print(similarity)
