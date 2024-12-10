import os
import json
import gradio as gr
from google.cloud import aiplatform
from google.oauth2 import credentials

# Get environment variables
project_id = os.getenv('PROJECT_ID')
endpoint_id = os.getenv('ENDPOINT_ID')
#location = os.getenv('REGION')
credentials_json = os.getenv('GOOGLE_CREDENTIALS')

if not all([project_id, endpoint_id, 'us-central1', credentials_json]):
    raise Exception("Missing required environment variables")

# Set up credentials and initialize Vertex AI
try:
    creds_dict = json.loads(credentials_json)
    credentials = credentials.Credentials.from_authorized_user_info(creds_dict)
    
    aiplatform.init(
        project=project_id,
        location='us-central1',
        credentials=credentials
    )
    
    endpoint = aiplatform.Endpoint(endpoint_id)
except Exception as e:
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

# Launch the app
port = int(os.getenv('PORT', 8080))
interface.launch(server_name="0.0.0.0", server_port=port)