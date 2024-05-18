from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
from duckdbRetriever import Retrieve
from pydantic import BaseModel
from pydantic import Field
import asyncio
from QAModel import QA, CustomBertForQuestionAnswering
import torch
from transformers import BertConfig, AutoTokenizer


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =================================== QA MODEL VAR ===================================
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
model.load_state_dict(torch.load('./QAModel/model.pth'), strict=False)
model_checkpoint = "atharvamundada99/bert-large-question-answering-finetuned-legal"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# =================================== REQUESTS ===================================

@app.get("/")                    
def read_items(): 
    return "Welcome to RAGn'Roll  :D <3 <3 <3 <3 :D"


class Query(BaseModel):
    query: str = Field(..., title="Query", example="What is the capital of France?")

class response(BaseModel):
    docs : List[str] = Field(..., example="['asdasd','gfdgdfg']")


@app.post("/retrieve", response_model= response)
async def retrieve(q: Query):
    try:
        # get the query from the body of the request
        query = q.query
        temp = Retrieve()
        result = temp.run(query)
        
        return {"docs": result}
    except Exception as e:
        return {"error": str(e)}
    

class QAReq(BaseModel):
    query: str = Field(..., title="Query", example="What is the capital of France?")
    doc: str = Field(..., title="Document", example= "Paris is the capital of France 3shan 5ater Hashish")

class responseQA(BaseModel):
    QA : str = Field(..., example="'lololololol'")

@app.post("/QAResponse", response_model= responseQA)
async def retrieve(q: QAReq):
    try:
        query = q.query
        doc = q.doc
        result = QA(model, tokenizer, query, doc)
        print(result)
        return {"QA": result}
    except Exception as e:
        return {"error": str(e)}