"""
OpenAPI Schema Export Utility for AGASTYA API.
Generates OpenAPI JSON schema for SDK generation, API clients, and documentation.
"""

import sys
import os
import json

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from services.api.app.main import app


def export_openapi_schema(output_path: str = "docs/openapi.json"):
    schema = app.openapi()
    full_output = os.path.join(BASE_DIR, output_path)
    os.makedirs(os.path.dirname(full_output), exist_ok=True)
    with open(full_output, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"Exported OpenAPI specification to {output_path}")


if __name__ == "__main__":
    export_openapi_schema()
