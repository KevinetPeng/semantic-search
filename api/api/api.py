from ninja import NinjaAPI, Schema
import cohere
import numpy as np
import os

cohere_key = os.getenv("COHERE_API_KEY")
cohere_model = os.getenv("COHERE_MODEL")

co = cohere.Client(cohere_key)

api = NinjaAPI()

class SearchSchema(Schema):
    query: str
    texts: list[str]


@api.get("/search")
def search(request, data: SearchSchema):
    #Encode your documents with input type 'search_document'
    doc_emb = co.embed(data.texts, input_type="search_document", model=cohere_model).embeddings
    doc_emb = np.asarray(doc_emb)

    #Encode your query with input type 'search_query'
    query_emb = co.embed([data.query], input_type="search_query", model=cohere_model).embeddings
    query_emb = np.asarray(query_emb)

    #Compute the dot product between query embedding and document embedding
    scores = np.dot(query_emb, doc_emb.T)[0]

    return scores.tolist()