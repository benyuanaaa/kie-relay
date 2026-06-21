#!/usr/bin/env python
"""Entry point for kie.ai relay server."""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
import uvicorn

if __name__ == "__main__":
    print(f">>> kie.ai relay starting on http://{settings.host}:{settings.port}")
    print(f"    KIE_API_BASE: {settings.kie_api_base}")
    key_status = 'configured' if settings.kie_api_key and settings.kie_api_key not in ('', 'your-kie-api-key-here') else 'NOT configured'
    print(f"    KIE_API_KEY:  {key_status}")
    print()
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=True,
    )
