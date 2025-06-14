from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import os
from typing import List, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import json
import asyncio
from datetime import datetime

app = FastAPI(title="AI Code Assistant", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel('gemini-pro')

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB
chroma_client = chromadb.Client()
try:
    collection = chroma_client.get_collection("code_knowledge")
except:
    collection = chroma_client.create_collection("code_knowledge")

class CodeRequest(BaseModel):
    query: str
    language: str = "vb.net"
    context: Optional[str] = None
    task_type: str = "generate"  # generate, debug, explain, optimize

class CodeResponse(BaseModel):
    code: str
    explanation: str
    suggestions: List[str]
    confidence: float

class RAGKnowledgeBase:
    def __init__(self):
        self.sample_knowledge = [
            {
                "id": "vb_net_basics",
                "content": "VB.NET basic syntax: Dim variable As DataType, Function/Sub declarations, If-Then-Else statements",
                "metadata": {"language": "vb.net", "topic": "basics"}
            },
            {
                "id": "onestream_brapi",
                "content": "OneStream BRApi common patterns: BRApi.Finance.Data.GetDataCell(), BRApi.ErrorLog.LogMessage()",
                "metadata": {"language": "vb.net", "topic": "brapi"}
            },
            {
                "id": "error_handling",
                "content": "VB.NET error handling: Try-Catch-Finally blocks, Throw statements, Exception types",
                "metadata": {"language": "vb.net", "topic": "error_handling"}
            }
        ]
        self.initialize_knowledge_base()
    
    def initialize_knowledge_base(self):
        """Initialize the vector database with sample knowledge"""
        for item in self.sample_knowledge:
            embedding = embedding_model.encode([item["content"]])[0].tolist()
            try:
                collection.add(
                    documents=[item["content"]],
                    embeddings=[embedding],
                    metadatas=[item["metadata"]],
                    ids=[item["id"]]
                )
            except:
                pass  # Skip if already exists
    
    def retrieve_relevant_context(self, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant context using RAG"""
        query_embedding = embedding_model.encode([query])[0].tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results["documents"][0] if results["documents"] else []

class AgenticCodeAssistant:
    def __init__(self):
        self.rag_kb = RAGKnowledgeBase()
        self.system_prompt = """You are an expert AI coding assistant specializing in VB.NET and OneStream development.
        Your capabilities include:
        1. Code generation with best practices
        2. Bug detection and fixing
        3. Code optimization suggestions
        4. OneStream BRApi integration help
        
        Always provide:
        - Clean, well-commented code
        - Clear explanations
        - Best practice recommendations
        - Security considerations
        """
    
    async def generate_code(self, request: CodeRequest) -> CodeResponse:
        """Main agentic workflow for code generation"""
        
        # Step 1: Retrieve relevant context using RAG
        relevant_context = self.rag_kb.retrieve_relevant_context(request.query)
        
        # Step 2: Build dynamic prompt based on task type
        prompt = self._build_prompt(request, relevant_context)
        
        # Step 3: Generate response with Gemini
        try:
            response = model.generate_content(prompt)
            
            # Step 4: Parse and structure the response
            parsed_response = self._parse_response(response.text, request.task_type)
            
            # Step 5: Self-correction check
            corrected_response = await self._self_correct(parsed_response, request)
            
            return corrected_response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")
    
    def _build_prompt(self, request: CodeRequest, context: List[str]) -> str:
        """Build dynamic prompt based on task type and context"""
        
        context_str = "\n".join(context) if context else "No specific context available."
        
        base_prompt = f"{self.system_prompt}\n\nRelevant Context:\n{context_str}\n\n"
        
        if request.task_type == "generate":
            task_prompt = f"Generate {request.language} code for: {request.query}"
        elif request.task_type == "debug":
            task_prompt = f"Debug and fix this {request.language} code: {request.query}"
        elif request.task_type == "explain":
            task_prompt = f"Explain this {request.language} code: {request.query}"
        elif request.task_type == "optimize":
            task_prompt = f"Optimize this {request.language} code: {request.query}"
        else:
            task_prompt = f"Help with this {request.language} request: {request.query}"
        
        if request.context:
            task_prompt += f"\n\nAdditional Context: {request.context}"
        
        full_prompt = f"""{base_prompt}
        
Task: {task_prompt}

Please provide your response in the following JSON format:
{{
    "code": "your generated/fixed code here",
    "explanation": "detailed explanation of the solution",
    "suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"],
    "confidence": 0.85
}}"""
        
        return full_prompt
    
    def _parse_response(self, response_text: str, task_type: str) -> CodeResponse:
        """Parse Gemini response into structured format"""
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                
                return CodeResponse(
                    code=parsed.get("code", ""),
                    explanation=parsed.get("explanation", ""),
                    suggestions=parsed.get("suggestions", []),
                    confidence=parsed.get("confidence", 0.7)
                )
            else:
                # Fallback parsing if JSON format is not found
                return CodeResponse(
                    code=response_text,
                    explanation="Generated response (parsing fallback)",
                    suggestions=["Review the generated code", "Test thoroughly"],
                    confidence=0.6
                )
                
        except Exception:
            return CodeResponse(
                code=response_text,
                explanation="Response generated with basic parsing",
                suggestions=["Verify code syntax", "Test functionality"],
                confidence=0.5
            )
    
    async def _self_correct(self, response: CodeResponse, original_request: CodeRequest) -> CodeResponse:
        """Self-correction mechanism to improve code quality"""
        
        correction_prompt = f"""Review and improve this generated code:

                                Original Request: {original_request.query}
                                Generated Code: {response.code}

                                Check for:
                                1. Syntax errors
                                2. Best practices
                                3. Security issues
                                4. Performance optimizations
                                5. OneStream specific patterns (if applicable)

                                Provide the corrected version in the same JSON format."""

        try:
            correction_response = model.generate_content(correction_prompt)
            corrected = self._parse_response(correction_response.text, original_request.task_type)
            
            # Increase confidence if corrections were made
            if corrected.code != response.code:
                corrected.confidence = min(corrected.confidence + 0.1, 1.0)
                corrected.suggestions.append("Code was self-corrected for improvements")
            
            return corrected
            
        except Exception:
            # Return original if correction fails
            return response

# Initialize the assistant
assistant = AgenticCodeAssistant()

@app.get("/")
async def root():
    return {"message": "AI Code Assistant API", "version": "1.0.0"}

@app.post("/generate", response_model=CodeResponse)
async def generate_code(request: CodeRequest):
    """Generate code based on natural language description"""
    return await assistant.generate_code(request)

@app.post("/debug", response_model=CodeResponse)
async def debug_code(request: CodeRequest):
    """Debug and fix provided code"""
    request.task_type = "debug"
    return await assistant.generate_code(request)

@app.post("/explain", response_model=CodeResponse)
async def explain_code(request: CodeRequest):
    """Explain provided code"""
    request.task_type = "explain"
    return await assistant.generate_code(request)

@app.post("/optimize", response_model=CodeResponse)
async def optimize_code(request: CodeRequest):
    """Optimize provided code"""
    request.task_type = "optimize"
    return await assistant.generate_code(request)

@app.post("/add_knowledge")
async def add_knowledge(content: str, metadata: dict):
    """Add new knowledge to the RAG database"""
    try:
        embedding = embedding_model.encode([content])[0].tolist()
        doc_id = f"doc_{datetime.now().timestamp()}"
        
        collection.add(
            documents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return {"message": "Knowledge added successfully", "id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add knowledge: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
