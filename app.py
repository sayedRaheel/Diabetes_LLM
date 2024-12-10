import os
from dotenv import load_dotenv
import gradio as gr
from google.cloud import aiplatform
from google.oauth2 import credentials
from google.auth.transport import requests
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# Load environment variables
load_dotenv()

# Get credentials from environment variables
email = os.getenv('GOOGLE_EMAIL')
password = os.getenv('GOOGLE_PASSWORD')
project_id = os.getenv('PROJECT_ID')
endpoint_id = os.getenv('ENDPOINT_ID')
location = os.getenv('REGION')

def get_auth_token():
    try:
        creds = Credentials(
            token=None,
            client_id=email,
            client_secret=password,
            token_uri="https://oauth2.googleapis.com/token",
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        
        creds.refresh(Request())
        return creds
    except Exception as e:
        print(f"Authentication failed: {str(e)}")
        raise Exception(f"Authentication failed: {str(e)}")

# Initialize Vertex AI with credentials
try:
    credentials = get_auth_token()
    aiplatform.init(
        project=project_id,
        location=location,
        credentials=credentials
    )
    endpoint = aiplatform.Endpoint(endpoint_id)
except Exception as e:
    print(f"Failed to initialize Vertex AI: {str(e)}")
    raise Exception(f"Failed to initialize Vertex AI: {str(e)}")

def predict(text):
    try:
        response = endpoint.predict(instances=[{"text": text}])
        return response.predictions[0]
    except Exception as e:
        return f"Error processing request: {str(e)}"

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Ask your question about diabetes..."
    ),
    outputs=gr.Textbox(label="Response"),
    title="Diabetica Medical Assistant",
    description="Ask questions about diabetes and get responses from our medical AI assistant.",
    examples=[
        ["What are the early symptoms of diabetes?"],
        ["How is Type 2 diabetes diagnosed?"],
        ["What lifestyle changes can help manage diabetes?"]
    ],
    theme="default"
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=int(os.getenv('PORT', 8084)))