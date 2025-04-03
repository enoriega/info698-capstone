import weaviate
from weaviate.auth import AuthApiKey
import json

# Initialize Weaviate client with API key
client = weaviate.Client(
    url="http://149.165.151.58:8080",
    auth_client_secret=AuthApiKey(api_key="ZEwUMscMSntgpraNQGU0EcBGXX5JccIgoJ0yk+k1sGE=")
)

# Get meta information about the instance
meta = client.get_meta()
print("Available modules and configuration:")
print(json.dumps(meta, indent=2))