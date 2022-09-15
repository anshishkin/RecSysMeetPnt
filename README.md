# RecSysMeetPnt

#update index FAISS /test
curl -H "Content-Type: application/json" -X POST http://0.0.0.0:8005/update_index -d @kuda_go_new.json

#test-query /test
curl -H "Content-Type: application/json" -X POST http://0.0.0.0:8005/query -d @kuda_query.json