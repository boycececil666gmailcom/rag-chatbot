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
