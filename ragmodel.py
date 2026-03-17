import faiss
from groq import Groq
import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

#API key setup
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def pdf_to_vectors(pdf_path):
    print("Reading PDF: ", pdf_path)
    with open(pdf_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        total_pages = len(pdf_reader.pages)
        #print(total_pages) 

        page_texts = []
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            page_texts.append({
                'text': page_text,
                'page_number': page_num+1
            })
        
        text = ''.join([p['text'] for p in page_texts])
    
    #print("Total pages: ", total_pages)
    #print("total text length: ", len(text))
    #print("Average characters per page: ", len(text)//total_pages)


    chunks=[]
    chunk_metadata =[]

    for i in range(0, len(text), 400):
        chunk_text = text[i:i+500]
        chunks.append(chunk_text)


        estimated_page = min((i// (len(text)//total_pages)) + 1, total_pages)
        chunk_metadata.append({
            'start_pos': i,
            'estimated_page': estimated_page
        })       
    print("created", len(chunks), "chunks")


    print("Getting embeddings locally...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = []
    for i, chunk in enumerate(chunks):
        print(f"Processing {i+1}/{len(chunks)}")
        embedding = model.encode(chunk)
        embeddings.append(embedding)
    embeddings = np.array(embeddings)

    print("Creating FAISS index...")
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]   
    index = faiss.IndexFlatL2(dimension)
    
    index.add(embeddings)
    
    print("Total vectors in index:", index.ntotal)
    
    faiss.write_index(index, "vectors.index")

    print("Saving to files.....")
    faiss.write_index(index, "vector.index")
    
    with open("chunks.pkl", "wb") as f:
        pickle.dump({
            'chunks': chunks,
            'metadata': chunk_metadata,
            'total_pages': total_pages
        }, f)
    print("vector database created successfully")
    return embeddings, chunks


if __name__ =="__main__":
    pdf_file = "KADIT'S Policy.pdf"
    embeddings, chunks = pdf_to_vectors(pdf_file)
    

    
   
