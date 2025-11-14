"""
vectordb_store.py
Pure Vector Database Storage - Imports chunks from chunks_dataset.py
"""

import chromadb
from chromadb.utils import embedding_functions
from chunks_dataset import chunk_markdown_files

class VectorDBStore:
    def __init__(self, persist_directory="./chroma_db"):
        """Initialize ChromaDB with embeddings"""
        
        # Create ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Setup embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Create collection (note: parameter is embedding_function, not embedding_functions)
        self.collection = self.client.get_or_create_collection(
            name="chatbot_knowledge",
            embedding_function=self.embedding_function
        )
        
        print(f"✓ VectorDB initialized: {persist_directory}")
        print(f"✓ Embedding model: all-MiniLM-L6-v2")
    
    def store_chunks(self, chunks):
        """Store chunks in VectorDB with embeddings"""
        
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            content = chunk.get('content', str(chunk))
            source = chunk.get('source', 'unknown')
            chunk_id = chunk.get('chunk_id', 0)
            
            documents.append(content)
            metadatas.append({
                'source': source,
                'chunk_index': chunk_id
            })
            ids.append(f"{source}_{chunk_id}")
        
        # Add to ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"✓ Stored {len(documents)} chunks in VectorDB")
        return len(documents)
    
    def query(self, question, n_results=3):
        """Query VectorDB for relevant chunks"""
        
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        return results
    
    def get_stats(self):
        """Get collection statistics"""
        count = self.collection.count()
        print(f"📊 Total chunks in VectorDB: {count}")
        return count


if __name__ == "__main__":
    
    print("="*80)
    print("VECTOR DATABASE STORAGE")
    print("="*80 + "\n")
    
    # Step 1: Import chunks from chunks_dataset.py
    print("Loading chunks from dataset...")
    dataset_path = "./chatbot_dataset"
    chunks = chunk_markdown_files(dataset_path)
    
    # Step 2: Initialize VectorDB
    print("\nInitializing VectorDB...")
    vectordb = VectorDBStore(persist_directory="./chroma_db")
    
    # Step 3: Store chunks with embeddings
    print("\nStoring chunks in VectorDB...")
    vectordb.store_chunks(chunks)
    
    # Step 4: Verify
    print("\n")
    vectordb.get_stats()
    
    # Step 5: Test query
    print("\n" + "="*80)
    print("TEST QUERY")
    print("="*80)
    
    question = "What services does Flowbotics offer?"
    print(f"\nQuestion: {question}\n")
    
    results = vectordb.query(question, n_results=3)
    
    for i, (doc, meta) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"[{i+1}] {meta['source']}")
        print(f"    {doc[:150]}...\n")
    
    print("="*80)
    print("✅ COMPLETE!")
    print("="*80)