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
import re

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
USE_QA = os.getenv("USE_QA", "false").lower() == "true"

# Pushover Configuration
PUSHOVER_USER_KEY = os.getenv("PUSHOVER_USER_KEY", "")
PUSHOVER_API_TOKEN = os.getenv("PUSHOVER_API_TOKEN", "")
ENABLE_PUSHOVER = os.getenv("ENABLE_PUSHOVER", "false").lower() == "true"

# Initialize S3 client if needed
if USE_S3 or USE_QA:
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
    """
    Searches the Q&A bucket using regex keywords and word matching.

    Scoring:
    - Keyword regex match: +10 points
    - Common words in the question: +1 point each

    Returns the top 3 best matches.
    """
    if not USE_QA or not QA_BUCKET:
        return []
    
    try:
        # Load all files from Q&A bucket
        response = s3_client.list_objects_v2(Bucket=QA_BUCKET)
        
        if 'Contents' not in response:
            print("No Q&A files found in bucket")
            return []
        
        qa_matches = []
        question_lower = question.lower()
        
        for obj in response['Contents']:
            if obj['Key'].endswith('.json'):
                # Download the contents of the file
                file_obj = s3_client.get_object(Bucket=QA_BUCKET, Key=obj['Key'])
                content = json.loads(file_obj['Body'].read().decode('utf-8'))

                match_score = 0
                keywords = content.get('keywords', [])
                matched_keywords = []

                for keyword_pattern in keywords:
                    try:
                        if re.search(keyword_pattern, question_lower, re.IGNORECASE):
                            match_score += 10  
                            matched_keywords.append(keyword_pattern)
                            print(f"✅ Keyword match: '{keyword_pattern}' in question: {question[:50]}...")
                    except re.error as e:
                        print(f"❌ Invalid regex pattern '{keyword_pattern}': {e}")
                        continue

                stored_question = content.get('question', '').lower()
                common_words = set(question_lower.split()) & set(stored_question.split())
                match_score += len(common_words)

                if match_score > 0:
                    qa_matches.append({
                        'question': content.get('question'),
                        'answer': content.get('answer'),
                        'category': content.get('category', 'general'),
                        'match_score': match_score,
                        'matched_keywords': matched_keywords
                    })
        
        qa_matches.sort(key=lambda x: x['match_score'], reverse=True)

        top_matches = qa_matches[:3]
        
        if top_matches:
            print(f"📚 Found {len(top_matches)} Q&A matches:")
            for i, match in enumerate(top_matches, 1):
                print(f"  {i}. Category: {match['category']}, Score: {match['match_score']}")
                print(f"     Matched keywords: {match['matched_keywords']}")
        
        return top_matches
        
    except Exception as e:
        print(f"❌ Error searching Q&A bucket: {e}")
        return []

def detect_unknown_answer(response: str) -> bool:
    """Detects if the model is unaware of responsibility"""
    unknown_phrases = [
        "i don't know",
        "nie wiem",
        "i'm not sure",
        r"\bi\s+(do\s+not|don't|dont)\s+have\b",
        "nie mam",
        "i cannot answer",
        "nie mogę odpowiedzieć",
        "i'm unable to",
        "nie jestem w stanie"
    ]

    response_lower = response.lower()
    return any(phrase in response_lower for phrase in unknown_phrases)

PERSONAL_INTENT_PATTERNS = [
    r"\bwhat are you interested in\b",
    r"\bwhat do you like\b",
    r"\bdo you enjoy\b",
    r"\byour (interests|hobbies|preferences|opinions)\b",
    r"\bwhat is your opinion\b",
]

def classify_intent(user_message: str) -> str:
    msg = user_message.lower()
    for pattern in PERSONAL_INTENT_PATTERNS:
        if re.search(pattern, msg):
            return "PERSONAL_PREFERENCE"
    return "SAFE"

def policy_decision(intent: str) -> Optional[str]:
    if intent == "PERSONAL_PREFERENCE":
        return (
            "I don't have enough information to answer that. "
            "Could you clarify or provide more details?"
        )
    return None

FORBIDDEN_OUTPUT_PHRASES = [
    "i enjoy",
    "my interests",
    "my hobbies",
    "as an ai",
    "built by amazon",
]

def validate_model_output(text: str) -> bool:
    text_lower = text.lower()
    return not any(p in text_lower for p in FORBIDDEN_OUTPUT_PHRASES)

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
    system_message = f"System: {prompt()}"
    
    # Add strict instructions when there is Q&A context
    if qa_context:
        system_message += """

    ========================================
    IMPORTANT INSTRUCTIONS FOR Q&A ANSWERS:
    ========================================

    You have been provided with VERIFIED answers from the knowledge base below.

    When answering questions that match the knowledge base:

    1. ✅ Use ONLY the information provided in the knowledge base
    2. ❌ Do NOT add any extra information or details not in the knowledge base
    3. ❌ Do NOT make up or hallucinate any information
    4. ✅ Answer in a natural, conversational way but stick STRICTLY to the facts provided
    5. ✅ If the knowledge base answer is incomplete, say: "Based on my knowledge base: [answer]. Would you like more details?"
    6. ✅ If multiple Q&A entries match, you can synthesize them but ONLY use the provided information

    """ + qa_context
    
    messages.append({
        "role": "user", 
        "content": [{"text": system_message}]
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
        temperature = 0.3 if qa_context else 0.7
        
        # Call Bedrock using the converse API
        response = bedrock_client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=messages,
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": temperature,  
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

        # === GUARDRAIL: pre-model check ===
        intent = classify_intent(request.message)
        policy_response = policy_decision(intent)

        if policy_response:
            # Do NOT call Bedrock
            conversation.append(
                {"role": "user", "content": request.message, "timestamp": datetime.now().isoformat()}
            )
            conversation.append(
                {
                    "role": "assistant",
                    "content": policy_response,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            save_conversation(session_id, conversation)

            return ChatResponse(
                response=policy_response,
                session_id=session_id
            )

        qa_results = search_qa_bucket(request.message)
        qa_context = ""

        if qa_results:
            qa_context = "=== VERIFIED KNOWLEDGE BASE ENTRIES ===\n\n"
            
            for i, qa in enumerate(qa_results, 1):
                qa_context += f"Entry {i} [Category: {qa['category']}]:\n"
                qa_context += f"Q: {qa['question']}\n"
                qa_context += f"A: {qa['answer']}\n"
                if qa.get('matched_keywords'):
                    qa_context += f"(Matched keywords: {', '.join(qa['matched_keywords'])})\n"
                qa_context += "\n"
            
            qa_context += "=== END OF KNOWLEDGE BASE ===\n"
            
            print(f"✅ Found {len(qa_results)} Q&A matches for: {request.message[:50]}...")
            print(f"📊 Top match: Category='{qa_results[0]['category']}', Score={qa_results[0]['match_score']}")

        # Call Bedrock for response
        assistant_response = call_bedrock(conversation, request.message, qa_context)
        if not validate_model_output(assistant_response):
            assistant_response = (
                "I don't have enough information to answer that accurately."
            )

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

        # Detect "I don't know" and send a Pushover notification
        if detect_unknown_answer(assistant_response):
            send_pushover_notification(request.message, session_id)
            print(f"Unknown question detected, Pushover notification sent: {request.message[:50]}...")

        return ChatResponse(
            response=assistant_response, 
            session_id=session_id,
            qa_matches=len(qa_results) 
        )

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