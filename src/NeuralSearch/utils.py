def print_documents(docs):
    for i, doc in enumerate(docs, 1):
        print(f"[{i}] {doc['title']}\n{doc['snippet']}\n{doc['link']}\n")