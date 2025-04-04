import weaviate
#from data_preprocess import weaviate_manager
from weaviate.auth import AuthApiKey
import os
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

# Define connection parameters
client = weaviate.connect_to_custom(
    http_host="149.165.151.58",
    http_port=8080,
    http_secure=False,  # Set to True if using HTTPS
    grpc_host="149.165.151.58",
    grpc_port=50051,    # Default gRPC port
    grpc_secure=False,  # Set to True if using secure gRPC
    auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY")) 
)

try:
    # Perform operations with the client
    if client.is_ready():
        print("Connected successfully!")
        # Add your operations here
finally:
    # Ensure the connection is closed
    client.close()
