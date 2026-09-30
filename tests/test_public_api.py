from llmrivotril import (
    BaseChunker,
    BaseLoader,
    BaseRetriever,
    MetricsCollector,
    RAGPipeline,
    __version__,
    global_metrics,
)


def test_public_api_exposes_extension_points_and_version():
    assert __version__ == "0.1.4"
    assert RAGPipeline is not None
    assert BaseLoader.__name__ == "BaseLoader"
    assert BaseChunker.__name__ == "BaseChunker"
    assert BaseRetriever.__name__ == "BaseRetriever"


def test_public_api_exposes_metrics_collector():
    # Regression test: MetricsCollector/global_metrics were only importable
    # via the llmrivotril.metrics submodule, not the top-level package, even
    # though RivotrilAgent(metrics=...) is documented as a public constructor
    # argument -- found while smoke-testing the published package.
    assert MetricsCollector is not None
    assert isinstance(global_metrics, MetricsCollector)
