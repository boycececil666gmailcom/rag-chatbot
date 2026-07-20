import pytest
from unittest.mock import patch

# Mock Embeddings class that runs locally without API keys
class MockEmbeddings:
    def __init__(self, *args, **kwargs):
        pass
    def embed_query(self, text):
        return [0.1] * 768
    def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]

# Start the patch globally so it applies to all imports during collection
patcher = patch("langchain_google_genai.GoogleGenerativeAIEmbeddings", MockEmbeddings)
patcher.start()

@pytest.fixture(autouse=True)
def reset_classifier_cache():
    try:
        import src.theme_based_rag_backend.agent_flow.nodes.classifier as classifier_module
        classifier_module.theme_embedding_cached = None
    except ImportError:
        pass
    yield
