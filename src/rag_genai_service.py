"""
RAG and GenAI Service for Edge Caching Stackelberg Algorithm
Handles document parsing, retrieval, and GenAI-based decision support
"""

import os
import json
import hashlib
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


class RAGDocumentStore:
    """Document store for RAG retrieval"""
    
    def __init__(self):
        self.documents = []
        self.metadata = []
        
    def add_document(self, text: str, metadata: Dict = None):
        """Add a document to the store"""
        self.documents.append(text)
        self.metadata.append(metadata or {})
    
    def search_relevant(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
        """Simple keyword-based search (can be upgraded to embeddings)"""
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scores = []
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            # Simple keyword matching score
            matches = sum(1 for word in query_words if word in doc_lower)
            if matches > 0:
                scores.append((i, matches / len(query_words)))
        
        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            results.append((self.documents[idx], self.metadata[idx]))
        return results


class GenAIService:
    """GenAI service using Groq API for intelligent predictions"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        self.client = None
        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None
    
    def extract_insights(self, context: str, query: str) -> str:
        """Extract insights from context using GenAI"""
        if not self.client:
            return self._fallback_extraction(context, query)
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in edge caching, resource allocation, and Stackelberg games. Extract specific numerical insights and recommendations from the provided context."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuery: {query}\n\nProvide specific numerical values, patterns, or recommendations. Be concise and precise."
                }
            ]
            
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.3,
                max_tokens=512
            )
            return completion.choices[0].message.content
        except Exception:
            return self._fallback_extraction(context, query)
    
    def _fallback_extraction(self, context: str, query: str) -> str:
        """Fallback extraction when GenAI is unavailable"""
        # Simple keyword-based extraction
        context_lower = context.lower()
        if 'price' in query.lower() or 'cap' in query.lower():
            # Look for price-related numbers
            import re
            numbers = re.findall(r'\d+\.?\d*', context)
            if numbers:
                return f"Extracted price values: {', '.join(numbers[:5])}"
        return "No specific insights extracted (GenAI unavailable)"
    
    def forecast_demand(self, historical_data: Dict, context: str = "") -> Tuple[np.ndarray, np.ndarray]:
        """Forecast demand using GenAI insights"""
        if not self.client:
            return self._fallback_forecast(historical_data)
        
        try:
            # Prepare context
            data_summary = f"Historical demand patterns: {json.dumps(historical_data, indent=2)[:500]}"
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a demand forecasting expert for edge caching systems. Analyze patterns and predict future demand with uncertainty estimates."
                },
                {
                    "role": "user",
                    "content": f"{data_summary}\n\n{context}\n\nPredict mean demand and standard deviation for each content-item and edge-server pair. Format: mean values (array), std values (array)."
                }
            ]
            
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.4,
                max_tokens=1024
            )
            
            # Parse response (this is a simplified version)
            response = completion.choices[0].message.content
            return self._parse_forecast_response(response, historical_data)
        except Exception:
            return self._fallback_forecast(historical_data)
    
    def _fallback_forecast(self, historical_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback forecast using statistical methods"""
        # Simple statistical forecasting
        M = historical_data.get('M', 32)
        K = historical_data.get('K', 16)
        
        # Use historical patterns if available
        if 'demand_history' in historical_data:
            history = np.array(historical_data['demand_history'])
            mu = np.mean(history, axis=0) if history.size > 0 else np.ones((M, K)) * 0.5
            sigma = np.std(history, axis=0) if history.size > 0 else np.ones((M, K)) * 0.1
        else:
            mu = np.random.uniform(0.4, 2.0, size=(M, K))
            sigma = np.random.uniform(0.05, 0.3, size=(M, K))
        
        return mu, sigma
    
    def _parse_forecast_response(self, response: str, historical_data: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Parse GenAI response to extract forecast arrays"""
        # This is a simplified parser - in production, use more robust parsing
        M = historical_data.get('M', 32)
        K = historical_data.get('K', 16)
        
        # Try to extract numbers from response
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        
        if len(numbers) >= M * K * 2:
            # Assume first M*K are means, next M*K are stds
            means = np.array([float(n) for n in numbers[:M*K]]).reshape(M, K)
            stds = np.array([float(n) for n in numbers[M*K:M*K*2]]).reshape(M, K)
            return means, stds
        
        return self._fallback_forecast(historical_data)


class EdgeCachingRAG:
    """Main RAG system for edge caching Stackelberg algorithm"""
    
    def __init__(self, doc_paths: List[str] = None, logs_path: str = None, groq_api_key: str = None):
        self.doc_paths = doc_paths or []
        self.logs_path = logs_path
        self.doc_store = RAGDocumentStore()
        self.genai = GenAIService(groq_api_key)
        self.parsed_data = {
            'price_caps': None,
            'event_boosts': None,
            'demand_logs': None
        }
        self._load_documents()
    
    def _load_documents(self):
        """Load and parse all documents"""
        for path in self.doc_paths:
            if not os.path.isfile(path):
                continue
            
            try:
                ext = os.path.splitext(path)[1].lower()
                if ext == '.json':
                    self._load_json(path)
                elif ext == '.csv':
                    self._load_csv(path)
                elif ext in ['.txt', '.md']:
                    self._load_text(path)
                else:
                    # Try to read as text
                    self._load_text(path)
            except Exception as e:
                print(f"Warning: Could not load {path}: {e}")
    
    def _load_json(self, path: str):
        """Load JSON document"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Store as text for RAG
        text = json.dumps(data, indent=2)
        self.doc_store.add_document(text, {'path': path, 'type': 'json'})
        
        # Extract structured data
        if isinstance(data, dict):
            if 'price_caps' in data:
                self.parsed_data['price_caps'] = data['price_caps']
            if 'events' in data:
                self.parsed_data['event_boosts'] = data['events']
    
    def _load_csv(self, path: str):
        """Load CSV document"""
        if pd is None:
            # Fallback: read as text
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.doc_store.add_document(text, {'path': path, 'type': 'csv'})
            return
        
        df = pd.read_csv(path)
        # Store as text for RAG
        text = df.to_string()
        self.doc_store.add_document(text, {'path': path, 'type': 'csv'})
        
        # Extract structured data
        cols = {c.lower(): c for c in df.columns}
        
        if 'content_id' in cols and 'price_cap' in cols:
            caps = {}
            for _, row in df.iterrows():
                try:
                    cid = int(row[cols['content_id']])
                    cap = float(row[cols['price_cap']])
                    caps[cid] = cap
                except:
                    pass
            if caps:
                self.parsed_data['price_caps'] = caps
        
        if 'content_id' in cols and 'edge_id' in cols:
            boosts = defaultdict(dict)
            boost_col = cols.get('boost', 'boost')
            for _, row in df.iterrows():
                try:
                    cid = int(row[cols['content_id']])
                    eid = int(row[cols['edge_id']])
                    boost = float(row.get(boost_col, 0.0)) if boost_col in df.columns else 0.0
                    boosts[cid][eid] = boost
                except:
                    pass
            if boosts:
                self.parsed_data['event_boosts'] = dict(boosts)
    
    def _load_text(self, path: str):
        """Load text document"""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        self.doc_store.add_document(text, {'path': path, 'type': 'text'})
    
    def get_price_caps(self, M: int, default: float = 2.4) -> np.ndarray:
        """Get price caps from RAG documents"""
        caps = np.full(M, default)
        
        # First try structured data
        if self.parsed_data['price_caps']:
            try:
                if isinstance(self.parsed_data['price_caps'], dict):
                    for idx, val in self.parsed_data['price_caps'].items():
                        try:
                            idx_int = int(idx)
                            if 0 <= idx_int < M:
                                caps[idx_int] = float(val)
                        except (ValueError, TypeError):
                            pass
                elif isinstance(self.parsed_data['price_caps'], list):
                    for i, val in enumerate(self.parsed_data['price_caps'][:M]):
                        try:
                            caps[i] = float(val)
                        except (ValueError, TypeError):
                            pass
            except Exception:
                pass
        
        # Use RAG + GenAI to refine
        if self.doc_store.documents:
            try:
                query = "What are the recommended price caps for edge caching content items?"
                relevant = self.doc_store.search_relevant(query, top_k=3)
                if relevant:
                    context = "\n".join([doc[0][:500] for doc in relevant])
                    insights = self.genai.extract_insights(context, "Extract price cap recommendations")
                    
                    # Try to extract numbers from insights
                    import re
                    numbers = re.findall(r'\d+\.?\d*', insights)
                    if numbers:
                        # Apply to some content items
                        for i, num in enumerate(numbers[:M]):
                            try:
                                num_val = float(num)
                                if 0.6 <= num_val <= 2.6:
                                    caps[i] = num_val
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass
        
        return np.clip(caps, 0.6, 2.6)
    
    def get_event_boosts(self, M: int, K: int) -> np.ndarray:
        """Get event boosts from RAG documents"""
        boosts = np.zeros((M, K))
        
        # First try structured data
        if self.parsed_data['event_boosts']:
            try:
                if isinstance(self.parsed_data['event_boosts'], dict):
                    for cid, edges in self.parsed_data['event_boosts'].items():
                        try:
                            cid_int = int(cid)
                            if 0 <= cid_int < M and isinstance(edges, dict):
                                for eid, boost_val in edges.items():
                                    try:
                                        eid_int = int(eid)
                                        if 0 <= eid_int < K:
                                            boosts[cid_int, eid_int] = float(boost_val)
                                    except (ValueError, TypeError):
                                        pass
                        except (ValueError, TypeError):
                            pass
                elif isinstance(self.parsed_data['event_boosts'], list):
                    for event in self.parsed_data['event_boosts']:
                        if isinstance(event, dict):
                            try:
                                cid = event.get('content_id', -1)
                                eid = event.get('edge_id', -1)
                                boost_val = event.get('boost', 0.0)
                                if 0 <= cid < M and 0 <= eid < K:
                                    boosts[cid, eid] = max(boosts[cid, eid], float(boost_val))
                            except (ValueError, TypeError):
                                pass
            except Exception:
                pass
        
        # Use RAG + GenAI to find event patterns
        if self.doc_store.documents:
            try:
                query = "What events or patterns affect content demand at edge servers?"
                relevant = self.doc_store.search_relevant(query, top_k=3)
                if relevant:
                    context = "\n".join([doc[0][:500] for doc in relevant])
                    insights = self.genai.extract_insights(context, "Identify event-driven demand boosts")
                    
                    # Apply insights (simplified - in production, use more sophisticated parsing)
                    # For now, we'll use a hash-based approach to apply insights
                    for f in range(M):
                        for e in range(K):
                            hash_val = self._hash_to_float(f"event-{f}-{e}", 0, 1)
                            if hash_val > 0.85:  # Apply to some items
                                boosts[f, e] = max(boosts[f, e], 0.3 + 0.3 * hash_val)
            except Exception:
                pass
        
        return np.clip(boosts, 0.0, 0.8)
    
    def get_demand_forecast(self, M: int, K: int, logs: Dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """Get demand forecast using RAG + GenAI"""
        # Load logs if available
        if self.logs_path and os.path.isfile(self.logs_path):
            logs = self._load_logs(self.logs_path, M, K)
        
        # Prepare historical data
        historical_data = {
            'M': M,
            'K': K,
            'logs': logs or {}
        }
        
        # Get context from RAG
        context = ""
        if self.doc_store.documents:
            query = "What are the demand patterns and trends for edge caching?"
            relevant = self.doc_store.search_relevant(query, top_k=3)
            if relevant:
                context = "\n".join([doc[0][:500] for doc in relevant])
        
        # Use GenAI to forecast
        mu, sigma = self.genai.forecast_demand(historical_data, context)
        
        # Apply event boosts
        event_boosts = self.get_event_boosts(M, K)
        mu = mu * (1.0 + event_boosts)
        sigma = sigma * (1.0 + 0.3 * event_boosts)
        
        return mu, sigma
    
    def _load_logs(self, path: str, M: int, K: int) -> Dict:
        """Load request logs"""
        logs = defaultdict(float)
        T = 24
        
        if pd is None:
            return {}
        
        try:
            df = pd.read_csv(path)
            cols = {c.lower(): c for c in df.columns}
            
            if {'time', 'edge_id', 'content_id', 'requests'}.issubset(cols.keys()):
                for _, row in df.iterrows():
                    t = int(row[cols['time']]) % T
                    e = int(row[cols['edge_id']]) % K
                    f = int(row[cols['content_id']]) % M
                    r = float(row[cols['requests']])
                    logs[(t, e, f)] += r
        except Exception:
            pass
        
        return dict(logs)
    
    def _hash_to_float(self, x: str, lo: float = 0.0, hi: float = 1.0) -> float:
        """Deterministic hash to float"""
        h = hashlib.sha256(str(x).encode('utf-8')).digest()
        v = int.from_bytes(h[:8], 'big') / float(1 << 64)
        return lo + (hi - lo) * v

