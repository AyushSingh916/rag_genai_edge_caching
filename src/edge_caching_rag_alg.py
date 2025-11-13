"""
Edge Caching RAG-GenAI Algorithm Implementation
This module provides the actual RAG-GenAI implementation for the Stackelberg algorithm.
It replaces the dummy simulate_rag and simulate_genai functions with real RAG-based logic.
"""

import numpy as np
import os
import json
from typing import Tuple, Optional
from rag_genai_service import EdgeCachingRAG

# Global RAG instance (initialized on first use)
_rag_instance = None
_doc_paths = []
_logs_path = None
_groq_api_key = None

_SCENARIO_CACHE = None
_SCENARIO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'edge_caching_case.json')


def _load_scenario():
    """Load bundled scenario data from JSON if available."""
    global _SCENARIO_CACHE
    if _SCENARIO_CACHE is None:
        path = os.path.abspath(_SCENARIO_PATH)
        if os.path.isfile(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    _SCENARIO_CACHE = json.load(f)
                print(f"Loaded scenario dataset from {path}")
            except Exception as e:
                print(f"Warning: Failed to read scenario dataset: {e}")
                _SCENARIO_CACHE = {}
        else:
            _SCENARIO_CACHE = {}
    return _SCENARIO_CACHE


def initialize_rag(doc_paths=None, logs_path=None, groq_api_key=None):
    """Initialize the RAG system with document paths and API key"""
    global _rag_instance, _doc_paths, _logs_path, _groq_api_key
    
    _doc_paths = doc_paths or []
    _logs_path = logs_path
    _groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY', None)
    
    if _doc_paths or _logs_path:
        try:
            _rag_instance = EdgeCachingRAG(
                doc_paths=_doc_paths,
                logs_path=_logs_path,
                groq_api_key=_groq_api_key
            )
            print(f"RAG system initialized with {len(_doc_paths)} document(s)")
        except Exception as e:
            print(f"Warning: Could not initialize RAG system: {e}")
            _rag_instance = None
    else:
        _rag_instance = None


def simulate_rag(M: int, K: int, boost_max: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get RAG-based event boosts and price caps.
    This replaces the dummy implementation with actual RAG retrieval.
    
    Args:
        M: Number of content items
        K: Number of edge servers
        boost_max: Maximum boost value (fallback if RAG unavailable)
    
    Returns:
        boost: Event boost matrix (M x K)
        cap: Price cap array (M,)
    """
    global _rag_instance
    
    # Initialize RAG if not already done
    if _rag_instance is None and (_doc_paths or _logs_path):
        initialize_rag(_doc_paths, _logs_path, _groq_api_key)
    
    if _rag_instance is not None:
        try:
            # Get actual boosts and caps from RAG
            boost = _rag_instance.get_event_boosts(M, K)
            cap = _rag_instance.get_price_caps(M, default=2.0)
            return boost, cap
        except Exception as e:
            print(f"Warning: RAG retrieval failed, using fallback: {e}")
    
    scenario = _load_scenario()
    if scenario:
        price_caps = scenario.get('price_caps')
        event_boosts = scenario.get('event_boosts')
        if price_caps:
            cap = np.full(M, 2.0)
            pc = np.array(price_caps, dtype=float)
            length = min(M, pc.shape[0])
            cap[:length] = pc[:length]
        else:
            cap = np.full(M, 2.0)
        if event_boosts:
            boost = np.zeros((M, K))
            eb = np.array(event_boosts, dtype=float)
            rows = min(M, eb.shape[0])
            cols = min(K, eb.shape[1])
            boost[:rows, :cols] = eb[:rows, :cols]
        else:
            boost = np.zeros((M, K))
        return boost, cap

    # Fallback to deterministic synthesis based on hash
    boost = np.zeros((M, K))
    cap = np.full(M, 2.0)
    
    # Deterministic hash-based generation (reproducible)
    import hashlib
    for f in range(M):
        h = hashlib.sha256(f"cap-{f}".encode()).digest()
        cap[f] = 1.6 + 0.8 * (int.from_bytes(h[:4], 'big') / (1 << 32))
        for e in range(K):
            h = hashlib.sha256(f"boost-{f}-{e}".encode()).digest()
            boost_val = boost_max * (int.from_bytes(h[:4], 'big') / (1 << 32))
            if boost_val > 0.85 * boost_max:  # Apply to some items
                boost[f, e] = boost_val
    
    return boost, cap


def simulate_genai(M: int, K: int, sigma_max: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get GenAI-based demand forecast (mean and std).
    This replaces the dummy implementation with actual GenAI forecasting.
    
    Args:
        M: Number of content items
        K: Number of edge servers
        sigma_max: Maximum standard deviation (fallback if GenAI unavailable)
    
    Returns:
        mu: Mean demand matrix (M x K)
        sigma: Standard deviation matrix (M x K)
    """
    global _rag_instance
    
    # Initialize RAG if not already done
    if _rag_instance is None and (_doc_paths or _logs_path):
        initialize_rag(_doc_paths, _logs_path, _groq_api_key)
    
    if _rag_instance is not None:
        try:
            # Get actual forecast from RAG + GenAI
            mu, sigma = _rag_instance.get_demand_forecast(M, K)
            # Ensure sigma is within bounds
            sigma = np.clip(sigma, 0.05, sigma_max)
            return mu, sigma
        except Exception as e:
            print(f"Warning: GenAI forecast failed, using fallback: {e}")
    
    scenario = _load_scenario()
    if scenario:
        mu_data = scenario.get('mu')
        sigma_data = scenario.get('sigma')
        mu = np.zeros((M, K))
        sigma = np.zeros((M, K))
        if mu_data:
            mu_arr = np.array(mu_data, dtype=float)
            rows = min(M, mu_arr.shape[0])
            cols = min(K, mu_arr.shape[1])
            mu[:rows, :cols] = mu_arr[:rows, :cols]
        if sigma_data:
            sigma_arr = np.array(sigma_data, dtype=float)
            rows = min(M, sigma_arr.shape[0])
            cols = min(K, sigma_arr.shape[1])
            sigma[:rows, :cols] = sigma_arr[:rows, :cols]
        sigma = np.clip(sigma, 0.05, sigma_max)
        mu = np.maximum(mu, 0.05)
        return mu, sigma

    # Fallback to statistical forecast
    # Use deterministic hash-based generation for reproducibility
    import hashlib
    mu = np.zeros((M, K))
    sigma = np.zeros((M, K))
    
    for f in range(M):
        for e in range(K):
            h = hashlib.sha256(f"mu-{f}-{e}".encode()).digest()
            mu_val = 0.4 + 1.6 * (int.from_bytes(h[:4], 'big') / (1 << 32))
            mu[f, e] = mu_val
            
            h = hashlib.sha256(f"sigma-{f}-{e}".encode()).digest()
            sigma_val = 0.05 + (sigma_max - 0.05) * (int.from_bytes(h[:4], 'big') / (1 << 32))
            sigma[f, e] = sigma_val
    
    return mu, sigma


# Convenience function to set configuration
def configure_rag(doc_paths=None, logs_path=None, groq_api_key=None):
    """Configure RAG system parameters"""
    global _doc_paths, _logs_path, _groq_api_key, _rag_instance
    
    _doc_paths = doc_paths or []
    _logs_path = logs_path
    _groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY', None)
    
    # Reinitialize if already initialized
    if _rag_instance is not None:
        initialize_rag(_doc_paths, _logs_path, _groq_api_key)

