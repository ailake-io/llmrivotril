from .agent import RivotrilAgent
from .exceptions import GuardrailViolationError, HallucinationDetectedError, LLMRivotrilError
from .guardrails import Guardrail
from .memory import MemoryStore
from .providers import (
    AnthropicProvider,
    BaseProvider,
    CohereProvider,
    GeminiProvider,
    OpenAIProvider,
    get_provider,
)
from .rag import Document, RAGPipeline
from .rag.chunkers import SimpleChunker
from .rag.loaders import MarkdownLoader, TextLoader
from .rag.retrievers import InMemoryEmbeddingRetriever, InMemoryKeywordRetriever
from .semantic import EmbeddingFaithfulnessVerifier, SemanticTopicGuardrail
from .verifier import ModelBasedFaithfulnessVerifier

__version__ = "0.5.0"
__all__ = [
    "RivotrilAgent",
    "Guardrail",
    "MemoryStore",
    "LLMRivotrilError",
    "GuardrailViolationError",
    "HallucinationDetectedError",
    "SemanticTopicGuardrail",
    "EmbeddingFaithfulnessVerifier",
    "ModelBasedFaithfulnessVerifier",
    "Document",
    "RAGPipeline",
    "TextLoader",
    "MarkdownLoader",
    "SimpleChunker",
    "InMemoryKeywordRetriever",
    "InMemoryEmbeddingRetriever",
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "CohereProvider",
    "GeminiProvider",
    "get_provider",
]
