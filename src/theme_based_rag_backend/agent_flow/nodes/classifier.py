import math
import logging
from src.theme_based_rag_backend.config import CHATBOT_THEME
from src.theme_based_rag_backend.agent_flow.state import AgentState

logger = logging.getLogger(__name__)

# Cached theme embedding to avoid redundant API calls
theme_embedding_cached = None

def get_theme_embedding(embeddings_model) -> list:
    global theme_embedding_cached
    if theme_embedding_cached is None:
        logger.info(f"Generating and caching embedding for CHATBOT_THEME: '{CHATBOT_THEME}'")
        theme_embedding_cached = embeddings_model.embed_query(CHATBOT_THEME)
    return theme_embedding_cached

def cosine_similarity(v1, v2) -> float:
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    norm_v2 = math.sqrt(sum(x * x for x in v2))
    if not norm_v1 or not norm_v2:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

def classifier_node(state: AgentState) -> dict:
    from src.theme_based_rag_backend.vector_db import get_vector_store
    import src.theme_based_rag_backend.vector_db as vector_db_module
    
    query = state["message"]
    
    print(f"\n\033[1;96m========================================================\033[0m")
    print(f"\033[1;92m>>> [Agent Flow] Classifying user query theme via Vector Similarity\033[0m")
    print(f"\033[1;96m========================================================\033[0m\n")
    
    try:
        # Ensure vector store is initialized to access the embeddings model
        get_vector_store()
        embeddings = vector_db_module.embeddings
        
        # Embed theme (cached) and query
        theme_vector = get_theme_embedding(embeddings)
        query_vector = embeddings.embed_query(query)
        
        # Calculate similarity
        similarity = cosine_similarity(theme_vector, query_vector)
        print(f"Cosine similarity between query and theme '{CHATBOT_THEME}': {similarity:.4f}")
        
        # Threshold check (0.65 is a good baseline for gemini-embedding-001)
        threshold = 0.65
        category = "rag" if similarity >= threshold else "refuse"
        
    except Exception as e:
        logger.error(f"Error during vector similarity classification: {e}. Falling back to 'refuse'.")
        category = "refuse"
        
    print(f"-> Categorized as: '{category}'")
    return {"category": category}

