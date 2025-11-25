#!/usr/bin/env python3
"""RAG System for WellGen AI - Retrieval Augmented Generation."""

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, knowledge_base_path="knowledge_base/kaggle_nutrition.json"):
        print("Initializing RAG system...")
        
        # Load embedding model
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        print("✓ Embedding model loaded")
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base(knowledge_base_path)
        print(f"✓ Loaded {len(self.knowledge_base)} documents from Kaggle nutrition datasets")
        
        # Create embeddings and index
        self._build_index()
        print("✓ RAG system ready")
    
    def _load_knowledge_base(self, path):
        """Load nutrition knowledge base from JSON."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _build_index(self):
        """Create FAISS index for fast similarity search."""
        # Extract text content from documents
        self.documents = []
        texts = []
        
        for doc in self.knowledge_base:
            text = f"{doc['title']}. {doc['content']}"
            texts.append(text)
            self.documents.append(doc)
        
        # Generate embeddings
        print("Generating embeddings...")
        self.embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
    
    def retrieve(self, query, top_k=3):
        """Retrieve most relevant documents for a query."""
        # Embed the query
        query_embedding = self.embedder.encode([query])
        
        # Search index
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Return relevant documents
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            doc = self.documents[idx]
            results.append({
                'title': doc['title'],
                'content': doc['content'],
                'category': doc['category'],
                'relevance_score': float(1 / (1 + distance))  # Convert distance to similarity
            })
        
        return results
    
    def augment_prompt(self, query, user_profile=None):
        """Retrieve relevant context and create augmented prompt."""
        # Retrieve relevant documents
        relevant_docs = self.retrieve(query, top_k=3)
        
        # Build context
        context = "RELEVANT NUTRITION KNOWLEDGE:\n\n"
        for i, doc in enumerate(relevant_docs, 1):
            context += f"{i}. {doc['title']}\n{doc['content']}\n\n"
        
        # Add user profile if available
        if user_profile:
            context += f"USER PROFILE:\n"
            context += f"- Age: {user_profile.get('age')}yo\n"
            context += f"- Gender: {user_profile.get('gender')}\n"
            context += f"- Height: {user_profile.get('height')}cm\n"
            context += f"- Weight: {user_profile.get('weight')}kg\n"
            context += f"- Goal: {user_profile.get('goal', 'weight_loss').replace('_', ' ').title()}\n"
            if user_profile.get('allergies') and user_profile['allergies'] != 'none':
                context += f"- Allergies: {user_profile['allergies']}\n"
            context += "\n"
        
        # Create augmented prompt
        augmented = f"""{context}USER QUESTION: {query}

INSTRUCTIONS:
- Use the provided nutrition knowledge to answer accurately
- Reference specific information from the context when relevant
- Be professional, encouraging, and personalized
- If user has allergies, ensure recommendations avoid those allergens
- Provide actionable, specific advice

RESPONSE:"""
        
        return augmented, relevant_docs

def test_rag():
    """Test the RAG system."""
    rag = RAGSystem()
    
    # Test queries
    test_queries = [
        "How much protein do I need to build muscle?",
        "What should I eat for diabetes?",
        "I'm allergic to nuts, what can I use instead?",
        "How many calories should I eat to lose weight?"
    ]
    
    print("\n" + "="*70)
    print("RAG SYSTEM TEST")
    print("="*70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        results = rag.retrieve(query, top_k=2)
        for i, doc in enumerate(results, 1):
            print(f"\n{i}. {doc['title']} (relevance: {doc['relevance_score']:.3f})")
            print(f"   {doc['content'][:150]}...")

if __name__ == "__main__":
    test_rag()
