import os
import sys
from fastapi.testclient import TestClient
from unittest.mock import patch

# Mock environment variables to satisfy imports
os.environ["CHATBOT_BACKEND_URL"] = "http://localhost:9999"
os.environ["GEMINI_API_KEY"] = "dummy_key"

def test_gateway_cors_wildcard():
    # If ALLOWED_ORIGINS is default or wildcard "*"
    with patch.dict(os.environ, {"ALLOWED_ORIGINS": "*"}):
        # Re-import or reload app to apply the environment variable
        import importlib
        import src.api_gateway.main
        importlib.reload(src.api_gateway.main)
        
        client = TestClient(src.api_gateway.main.app)
        
        # Test OPTIONS preflight request
        response = client.options(
            "/health",
            headers={
                "Origin": "https://example.com",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            }
        )
        assert response.status_code == 200
        # When ALLOWED_ORIGINS is "*", Starlette returns "*" or the origin depending on credentials.
        # Since allow_credentials=False for wildcard, access-control-allow-origin will be "*"
        assert response.headers.get("access-control-allow-origin") == "*"
        assert "access-control-allow-credentials" not in response.headers

def test_gateway_cors_specific_origins():
    # If ALLOWED_ORIGINS is set to a specific list of domains
    with patch.dict(os.environ, {"ALLOWED_ORIGINS": "https://my-app.com,https://dashboard.my-app.com"}):
        import importlib
        import src.api_gateway.main
        importlib.reload(src.api_gateway.main)
        
        client = TestClient(src.api_gateway.main.app)
        
        # Test preflight from an allowed origin
        response = client.options(
            "/health",
            headers={
                "Origin": "https://my-app.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            }
        )
        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "https://my-app.com"
        assert response.headers.get("access-control-allow-credentials") == "true"
        
        # Test preflight from a disallowed origin
        response_bad = client.options(
            "/health",
            headers={
                "Origin": "https://malicious-site.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            }
        )
        # CORS middleware does not reject preflight with non-200, it just doesn't return CORS headers
        assert "access-control-allow-origin" not in response_bad.headers
