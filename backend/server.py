from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from typing import Optional, List, Dict
import json
import uuid
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
from context import prompt
import requests

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize Bedrock client
bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=os.getenv("DEFAULT_AWS_REGION", "eu-west-1")
)

# Bedrock model selection
# Available models:
# - amazon.nova-micro-v1:0  (fastest, cheapest)
# - amazon.nova-lite-v1:0   (balanced - default)
# - amazon.nova-pro-v1:0    (most capable, higher cost)
# Remember the Heads up: you might need to add us. or eu. prefix to the below model id
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "eu.amazon.nova-micro-v1:0")

# Memory storage configuration
USE_S3 = os.getenv("USE_S3", "false").lower() == "true"
S3_BUCKET = os.getenv("S3_BUCKET", "")
MEMORY_DIR = os.getenv("MEMORY_DIR", "../memory")

# Q&A Configuration
QA_BUCKET = os.getenv("QA_BUCKET", "")
USE_QA = os.getenv("USE_QA", "false").lower() = "true"

# Pushover Configuration
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")
PUSHOVER_API_TOKEN = os.getenv("PUSHOVER_API_TOKEN", "")
ENABLE_PUSHOVER = os.getenv("ENABLE_PUSHOVER", "false").lower() == "true"

# Initialize S3 client if needed
if USE_S3:
    s3_client = boto3.client("s3")


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    session_id: str


class Message(BaseModel):
    role: str
    content: str
    timestamp: str

def search_qa_bucket(question: str) -> List[Dict]:
    """Searches through the question and answer bucket for the right answers"""
    if not USE_QA or not QA_BUCKET:
        return []

    try:
        response = s3_client.list_objects_v2(Bucket=QA_BUCKET)

        if 'Contents' not in response:
            print("Not Q&A files in bucket")
            return []

        qa_pairs = []
        question_lower = question.lower()

        for obj in response['Contents']:
            if obj['Key'].endswith('.json'):
                file_obj = s3_client.get_object(Bucket=QA_BUCKET, Key=obj['Key'])
                content = json.loads(file_obj['Body'].read().decode('utf-8'))

                stored_question = content.get('question', '').lower()

                if any(word in stored_question for word in question_lower.split() if len(word) > 3):
                    qa_pairs.append({
                        'question': content.get('question'),
                        'answer': content.get('answer'),
                        'relevance': len(set(question_lower.split()) & set(stored_question.split()))
                    })
        qa_pairs.sort(key=lambda x: x['relevance'], reverse=True)
        return qa_pairs[:3]
    except Exception as e:
        print(f"Error searching Q&A bucket: {e}")
        return []

def detect_unknown_answer(response: str) -> bool:
    """Detects if the model is unaware of responsibility"""
    unknown_phrases = [
        "i don't know",
        "nie wiem",
        "i'm not sure",
        "nie jestem pewien",
        "i don't have information",
        "nie mam informacji",
        "i cannot answer",
        "nie mogę odpowiedzieć",
        "i'm unable to",
        "nie jestem w stanie"
    ]

    response_lower = response.lower()
    return any(phrase in response_lower for phrase in unknown_phrases)

def send_pushover_notification(question: str, session_id: str):
    """Sends a Pushover notification about an unknown question"""
    if not ENABLE_PUSHOVER or not PUSHOVER_USER_KEY or not PUSHOVER_API_TOKEN:
        print("Pushover not configured or disabled")
        return

    try:
        message = f"Unknown question in Digital Twin\n\n{question}\n\nSession: {session_id}"
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": PUSHOVER_API_TOKEN,
                "user": PUSHOVER_USER_KEY,
                "message": message,
                "title": "Digital Twin - Unknown Question",
                "priority": 0
            },
            timeout=10
        )

        if response.status_code == 200:
            print(f"Pushover notification sent sucessfully for question: {question[:50]}..")
        else:
            print(f"Failed to send Pushover notification: {response.status_code}")
    except Exception as e:
        print(f"Error sending Pushover notification: {e}")

# Memory management functions
def get_memory_path(session_id: str) -> str:
    return f"{session_id}.json"


def load_conversation(session_id: str) -> List[Dict]:
    """Load conversation history from storage"""
    if USE_S3:
        try:
            response = s3_client.get_object(Bucket=S3_BUCKET, Key=get_memory_path(session_id))
            return json.loads(response["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                return []
            raise
    else:
        # Local file storage
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return json.load(f)
        return []


def save_conversation(session_id: str, messages: List[Dict]):
    """Save conversation history to storage"""
    if USE_S3:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=get_memory_path(session_id),
            Body=json.dumps(messages, indent=2),
            ContentType="application/json",
        )
    else:
        # Local file storage
        os.makedirs(MEMORY_DIR, exist_ok=True)
        file_path = os.path.join(MEMORY_DIR, get_memory_path(session_id))
        with open(file_path, "w") as f:
            json.dump(messages, f, indent=2)


def call_bedrock(conversation: List[Dict], user_message: str, qa_context: str = "") -> str:
    """Call AWS Bedrock with conversation history"""
    
    # Build messages in Bedrock format
    messages = []
    
    # Add system prompt as first user message (Bedrock convention)
    messages.append({
        "role": "user", 
        "content": [{"text": f"System: {prompt()}"}]
    })
    
    # Add conversation history (limit to last 10 exchanges to manage context)
    for msg in conversation[-20:]:  # Last 10 back-and-forth exchanges
        messages.append({
            "role": msg["role"],
            "content": [{"text": msg["content"]}]
        })
    
    # Add current user message
    messages.append({
        "role": "user",
        "content": [{"text": user_message}]
    })
    
    try:
        # Call Bedrock using the converse API
        response = bedrock_client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=messages,
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 0.7,
                "topP": 0.9
            }
        )
        
        # Extract the response text
        return response["output"]["message"]["content"][0]["text"]
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException':
            # Handle message format issues
            print(f"Bedrock validation error: {e}")
            raise HTTPException(status_code=400, detail="Invalid message format for Bedrock")
        elif error_code == 'AccessDeniedException':
            print(f"Bedrock access denied: {e}")
            raise HTTPException(status_code=403, detail="Access denied to Bedrock model")
        else:
            print(f"Bedrock error: {e}")
            raise HTTPException(status_code=500, detail=f"Bedrock error: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "AI Digital Twin API (Powered by AWS Bedrock)",
        "memory_enabled": True,
        "storage": "S3" if USE_S3 else "local",
        "ai_model": BEDROCK_MODEL_ID,
        "qa_enabled": USE_QA,
        "pushover_enabled": ENABLE_PUSHOVER
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "use_s3": USE_S3,
        "bedrock_model": BEDROCK_MODEL_ID,
        "qa_enabled": USE_QA,
        "pushover_enabled": ENABLE_PUSHOVER
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())

        # Load conversation history
        conversation = load_conversation(session_id)

        qa_results = search_qa_bucket(request.message)
        qa_context = ""

        if qa_results:
            qa_context = "I found the following information in the knowledge base:"
            for i, qa in enumerate(qa_results, 1):
                qa_context += f"{i}. Question: {qa['question']}\n   Answer: {qa['answer']}\n\n"

            print(f"Found {len(qa_results)} Q&A matches for {request.message[:50]}...")

        assistant_response = call_bedrock(conversation, request.message, qa_context)

        # Call Bedrock for response
        assistant_response = call_bedrock(conversation, request.message)

        # Update conversation history
        conversation.append(
            {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()}
        )
        conversation.append(
            {
                "role": "assistant",
                "content": assistant_response,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Save conversation
        save_conversation(session_id, conversation)

        if detect_unknown_answer(assistant_response):
            send_pushover_notification(request.message, session_id)
            print(f"Unknown question detected, Pushover notification sent: {request.message[:50]}...")

        return ChatResponse(response=assistant_response, session_id=session_id, qa_matches=len(qa_results))

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/{session_id}")
async def get_conversation(session_id: str):
    """Retrieve conversation history"""
    try:
        conversation = load_conversation(session_id)
        return {"session_id": session_id, "messages": conversation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)