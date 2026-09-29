from llmrivotril import (
    BaseChunker,
    BaseLoader,
    BaseRetriever,
    RAGPipeline,
    __version__,
)


def test_public_api_exposes_extension_points_and_version():
    assert __version__ == "0.1.0"
    assert RAGPipeline is not None
    assert BaseLoader.__name__ == "BaseLoader"
    assert BaseChunker.__name__ == "BaseChunker"
    assert BaseRetriever.__name__ == "BaseRetriever"
