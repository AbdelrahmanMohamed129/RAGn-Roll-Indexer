from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from duckdbRetriever import Retrieve
import duckdbRetriever
from pydantic import BaseModel
from pydantic import Field
from QAModel import QA, CustomBertForQuestionAnswering
import torch
from transformers import  DPRQuestionEncoder, DPRQuestionEncoderTokenizer
from transformers import AutoTokenizer, pipeline, BertConfig
import clip
from PIL import Image, PngImagePlugin
import os
from fastapi import FastAPI
import uvicorn

import sys
sys.path.insert(1, './QACode')


from RagNRollQA import RagNRollQA


os.environ['KMP_DUPLICATE_LIB_OK']='True'

Image.MAX_IMAGE_PIXELS = None
PngImagePlugin.MAX_TEXT_CHUNK = 1024*1024*10

device = "cuda" if torch.cuda.is_available() else "cpu"

# =================================== INDEXER MODELS ===================================
tokenizerText = DPRQuestionEncoderTokenizer.from_pretrained('Z:/Data/Model/')
modelText = DPRQuestionEncoder.from_pretrained('Z:/Data/Model/')
modelImg = clip.load('ViT-L/14', device=device)
# modelImg,_ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')


# =================================== QA MODELS ===================================
# Instantiate the model with the provided configuration
config = BertConfig.from_dict({
    "_name_or_path": "ourModel",
    "architectures": [
        "BertForQuestionAnswering"
    ],
    "attention_probs_dropout_prob": 0.1,
    "gradient_checkpointing": False,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "hidden_size": 1024,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "bert",
    "num_attention_heads": 16,
    "num_hidden_layers": 24,
    "pad_token_id": 0,
    "position_embedding_type": "absolute",
    "transformers_version": "4.17.0",
    "type_vocab_size": 2,
    "use_cache": True,
    "vocab_size": 30522
})
model = CustomBertForQuestionAnswering(config)
model.load_state_dict(torch.load('./QAModel/CustomModel.pth'), strict=False)
model_checkpoint = "atharvamundada99/bert-large-question-answering-finetuned-legal"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

question_answerer = pipeline("question-answering", model=model, tokenizer=tokenizer)


RAGnRoll = RagNRollQA()

# =================================== REQUESTS ===================================
app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")                    
def read_items(): 
    return "Welcome to RAGn'Roll  :D <3 <3 <3 <3 :D"


class Query(BaseModel):
    query: str = Field(..., title="Query", example="What is the capital of France?")
class TextIds(BaseModel):
    query: str = Field(..., title="Query", example="What is the capital of France?")
    ids : List[int] = Field(..., example="[1, 2]")

class response(BaseModel):
    docs : List[str] = Field(..., example=['asdasd','gfdgdfg'])
class responseIds(BaseModel):
    ids : List[int] = Field(..., example="[1, 2]")
class responseURLs(BaseModel):
    urls : List[dict] = Field(..., example=[{'link':'asdasdasdasd'}])


@app.post("/retrieveIdsText", response_model= responseIds)
def retrieveIdsText(q: Query):
    try:
        # get the query from the body of the request
        query = q.query
        temp = Retrieve()
        result = temp.returnIds(query, isImage= False, modelText= modelText, tokenizerText= tokenizerText)
        
        return {"ids": result}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/retrieveActualText")
def retrieveActualText(q: TextIds):
    try:
        # get the query from the body of the request
        ids = q.ids
        query = q.query
        temp = Retrieve()
        result = temp.returnActualData(ids, query)
        
        return {"docs": result}
    except Exception as e:
        return {"error": str(e)}
    
@app.post("/retrieveImgs", response_model= responseURLs)
def retrieveImgs(q: Query):
    try:
        # get the query from the body of the request
        query = q.query
        temp = Retrieve()
        result = temp.run(query, isImage=True, modelImg= modelImg, device= device)
        # convert this array to array of dicts where each one has key link and value its url
        result = [{"link": i} for i in result]
        return {"urls": result}
    except Exception as e:
        return {"error": str(e)}
    

class QAReq(BaseModel):
    query: str = Field(..., title="Query", example="What is the capital of France?")
    doc: str = Field(..., title="Document", example= "Paris is the capital of France 3shan 5ater Hashish")
    ourModel: bool = Field(..., title="OurModel", example= True)

class responseQA(BaseModel):
    QA : str = Field(..., example="'lololololol'")

@app.post("/QAResponse", response_model= responseQA)
def retrieve(q: QAReq):
    try:
        query = q.query
        doc = q.doc
        check = q.ourModel

        if check:
            result = RAGnRoll.answer_question(query, doc)
        else:
            # result = QA(model, tokenizer, query, doc)
            result = question_answerer(question=query, context=doc, truncation=True, padding=True, return_tensors='pt')
            result = result["answer"]
        

        return {"QA": result}
    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run("Deployment:app", host="127.0.0.1", port=3001, log_level="info")