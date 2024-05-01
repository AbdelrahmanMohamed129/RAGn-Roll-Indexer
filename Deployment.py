from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np
from duckdbRetriever import Retrieve
from pydantic import BaseModel
from pydantic import Field
import asyncio

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

# Input Schema
class Query(BaseModel):
    query: str = Field(..., title="Query", example="What is the capital of France?")

# Output Schema
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