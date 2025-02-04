from .bm25 import BM25
from .colbertv2 import Colbertv2
from .llm_reranker import LLMReranker
from .multi_vss import MultiVSS
from .vss import  VSS
from .papr import Papr


REGISTERED_MODELS = [
    'BM25', 
    'Colbertv2', 
    'VSS', 
    'MultiVSS', 
    'LLMReranker',
    'Paprv1'
]
