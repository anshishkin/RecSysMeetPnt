from flask import Flask, jsonify, request
from navec import Navec

import os
from math import sin, cos, sqrt, atan2, radians

import string
from typing import Dict, List, Tuple, Union, Callable

import nltk
import numpy as np
import json
import faiss


path = 'additional_data/navec_news_v1_1B_250K_300d_100q.tar'

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

index_is_ready = False

class Helper:
    def __init__(self):
        self.navec=Navec.load(path)

    def _hadle_punctuation(self, inp_str: str) -> str:
        inp_str = str(inp_str)
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct, ' ')
        return inp_str
    
    def _simple_preproc(self, inp_str: str):
        base_str = inp_str.strip().lower()
        str_wo_punct = self._hadle_punctuation(base_str)
        return nltk.word_tokenize(str_wo_punct)

    
    def distance(self,lat1,lon1,lat2,lon2):
        R = 6373.0

        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def prepare_index(self, documents):
        d=dict()
        locn=dict()
        geo=dict()
        for i in documents:
            d[i['id']]=i['title']
            locn[i['id']]=i['location']
            geo[i['id']]=i['geo']
           
        self.location=locn
        self.geo=geo    
        self.documents = d
        idxs, docs, = [], []
        for idx in d:
            idxs.append(int(idx))
            docs.append(d[idx])
        embeddings = []
        for d in docs:
            tmp_emb= [w if w in self.navec else str('<unk>') for w in self._simple_preproc(d)]
            emb=np.empty(shape=[0,300])
            for i in tmp_emb:
                emb=np.vstack((emb,self.navec[i]))
            tmp_emb = np.mean(emb,axis=0)
            embeddings.append(np.array(tmp_emb))          
        embeddings = np.array([embedding for embedding in embeddings]).astype(np.float32)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(embeddings, np.array(idxs))
        index_size = self.index.ntotal
        global index_is_ready
        index_is_ready = True
        print(index_is_ready)
        return index_size

    def _filter(self, place:str,geo_user:list, radius:float=3.0):
        dict_={key: value for key, value in self.geo.items() if self.distance(value[0],value[1],geo_user[0],geo_user[1])>=radius} 
        idxs_=[x for x, v in dict_.items()]
        idxs=[x for x, v in self.location.items() if v!=place] 
        uniques = np.unique(np.append(idxs_,idxs))
        index=self.index
        index.remove_ids(np.array(uniques,np.int64))
        return index
    
    def get_suggestion(self, 
            query: str, 
            index,
            ann_k: int = 3) -> List[Tuple[str, str]]:
        q_tokens = self._simple_preproc(query)
        vector = [ tok if tok in self.navec else str('<unk>') for tok in q_tokens]
        emb=np.empty(shape=[0,300])
        for i in vector:
            emb=np.vstack((emb,self.navec[i]))
        q_emb = np.mean(emb,axis=0).reshape(1, -1)
        q_emb = np.array(q_emb).astype(np.float32)

        _, I = index.search(q_emb, k = ann_k)
        cands = [(str(i), self.documents[str(i)]) for i in I[0] if i != -1]
        return cands

    def query_handler(self, inp):
        #input_json =json.loads(inp.json)
        input_json = inp.json
        queries = input_json["queries"]
        suggestions = []
        index=self._filter(queries["location"],queries["geo"])
        for q in queries["topic"]:
            suggestion = self.get_suggestion(q,index)
            suggestions.append(suggestion)
        return suggestions

    def index_handler(self, inp):
        #input_json =json.loads(inp.json)
        input_json = inp.json
        documents = input_json["documents"]
        index_size = self.prepare_index(documents)
        return index_size

hlp = Helper()

@app.route('/ping')
def ping():
    if not hlp:
        return jsonify(status="not ready")
    return jsonify(status="ok")

@app.route('/query', methods=['POST'])
def query():
    if not index_is_ready:
        return json.dumps({"status": "FAISS is not initialized!"})
    suggestions = hlp.query_handler(request)

    return jsonify(suggestions=suggestions)

@app.route('/update_index', methods=['POST'])
def update_index():
    index_size = hlp.index_handler(request)

    return jsonify(status="ok", index_size=index_size)

