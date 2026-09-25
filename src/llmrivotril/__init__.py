from .agent import RivotrilAgent
from .exceptions import (
    GuardrailViolationError,
    HallucinationDetectedError,
    LLMRivotrilError,
    TokenBudgetExceededError,
)
from .guardrails import Guardrail
from .memory import MemoryStore
from .plugins import discover_plugins, load_plugins
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
from .rag.loaders import CSVLoader, HTMLLoader, MarkdownLoader, PDFLoader, TextLoader
from .rag.retrievers import InMemoryEmbeddingRetriever, InMemoryKeywordRetriever
from .semantic import EmbeddingFaithfulnessVerifier, SemanticTopicGuardrail
from .tools import ToolCall, ToolRegistry
from .verifier import ModelBasedFaithfulnessVerifier

__version__ = "0.5.0"
__all__ = [
    "RivotrilAgent",
    "Guardrail",
    "MemoryStore",
    "LLMRivotrilError",
    "GuardrailViolationError",
    "HallucinationDetectedError",
    "TokenBudgetExceededError",
    "discover_plugins",
    "load_plugins",
    "ToolRegistry",
    "ToolCall",
    "SemanticTopicGuardrail",
    "EmbeddingFaithfulnessVerifier",
    "ModelBasedFaithfulnessVerifier",
    "Document",
    "RAGPipeline",
    "TextLoader",
    "MarkdownLoader",
    "HTMLLoader",
    "CSVLoader",
    "PDFLoader",
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
