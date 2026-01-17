"""
Generation Module with Amazon Bedrock Nova Lite
"""

import json
from typing import List, Generator, Optional, Dict, Any

import boto3
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_aws import ChatBedrockConverse

from src.config import config


class ResponseGenerator:
    """
    Response generation using Amazon Bedrock Nova Lite model
    """
    
    def __init__(self):
        self.model_id = config.aws.bedrock_model_id
        self.region = config.aws.region
        
        # LangChain Bedrock client for structured interactions
        self.llm = ChatBedrockConverse(
            model=self.model_id,
            region_name=self.region,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature
        )
        
        # Raw Bedrock client for streaming
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=self.region
        )
    
    def generate(
        self, 
        query: str, 
        context_docs: List[Document],
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response based on query and retrieved context
        """
        # Build context from documents
        context = self._format_context(context_docs)
        
        # Build system prompt
        full_system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Build messages
        messages = [SystemMessage(content=full_system_prompt)]
        
        # Add chat history
        if chat_history:
            for msg in chat_history:
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
                else:
                    messages.append(AIMessage(content=msg['content']))
        
        # Add current query with context
        user_message = f"""Context from documents:
{context}

---

User Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain relevant information, say so clearly."""

        messages.append(HumanMessage(content=user_message))
        
        # Generate response
        response = self.llm.invoke(messages)
        
        return response.content
    
    def generate_stream(
        self, 
        query: str, 
        context_docs: List[Document],
        chat_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Generate a streaming response
        """
        # Build context
        context = self._format_context(context_docs)
        full_system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Build messages for Bedrock
        messages = []
        
        # Add chat history
        if chat_history:
            for msg in chat_history:
                messages.append({
                    "role": msg['role'],
                    "content": [{"text": msg['content']}]
                })
        
        # Add current query with context
        user_message = f"""Context from documents:
{context}

---

User Question: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't contain relevant information, say so clearly."""

        messages.append({
            "role": "user",
            "content": [{"text": user_message}]
        })
        
        # Prepare request
        request_body = {
            "messages": messages,
            "system": [{"text": full_system_prompt}],
            "inferenceConfig": {
                "maxTokens": config.model.max_tokens,
                "temperature": config.model.temperature
            }
        }
        
        # Stream response
        response = self.bedrock_client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        # Process stream
        for event in response['body']:
            chunk = json.loads(event['chunk']['bytes'])
            
            if 'contentBlockDelta' in chunk:
                delta = chunk['contentBlockDelta'].get('delta', {})
                if 'text' in delta:
                    yield delta['text']
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string"""
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', 'N/A')
            doc_type = doc.metadata.get('doc_type', 'text')
            score = doc.metadata.get('relevance_score', 'N/A')
            
            header = f"[Document {i}] Source: {source}, Page: {page}, Type: {doc_type}"
            if score != 'N/A':
                header += f", Relevance: {score:.2f}"
            
            context_parts.append(f"{header}\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for RAG"""
        return """You are an intelligent assistant powered by a Retrieval Augmented Generation (RAG) system. Your role is to answer questions based on the provided document context.

Guidelines:
1. Base your answers primarily on the provided context
2. If the context contains tables, interpret and present the data clearly
3. If the context includes formulas or equations, explain them when relevant
4. Cite specific documents or page numbers when making claims
5. If you cannot find the answer in the context, clearly state that
6. Be concise but comprehensive
7. Use markdown formatting for better readability

Remember: Accuracy is paramount. Never fabricate information not present in the context."""


class QueryAnalyzer:
    """
    Analyzes user queries to enhance retrieval
    """
    
    def __init__(self):
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=config.aws.region
        )
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query into multiple search variants for better retrieval
        """
        prompt = f"""Given the following user question, generate 3 alternative phrasings or related queries that could help find relevant information. Return only the queries, one per line.

Original question: {query}

Alternative queries:"""

        request_body = {
            "messages": [
                {"role": "user", "content": [{"text": prompt}]}
            ],
            "inferenceConfig": {
                "maxTokens": 200,
                "temperature": 0.7
            }
        }
        
        response = self.bedrock_client.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        text = response_body['output']['message']['content'][0]['text']
        
        # Parse queries
        queries = [query]  # Include original
        for line in text.strip().split('\n'):
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove numbering if present
                if line[0].isdigit() and '.' in line[:3]:
                    line = line.split('.', 1)[1].strip()
                queries.append(line)
        
        return queries[:4]  # Return max 4 queries
    
    def classify_query_type(self, query: str) -> Dict[str, Any]:
        """
        Classify the type of query to optimize retrieval strategy
        """
        prompt = f"""Analyze the following query and classify it. Respond in JSON format with these fields:
- "type": one of ["factual", "analytical", "comparison", "procedural", "definition"]
- "needs_tables": boolean, true if likely needs tabular data
- "needs_visuals": boolean, true if likely needs image/diagram info
- "complexity": one of ["simple", "moderate", "complex"]

Query: {query}

JSON response:"""

        request_body = {
            "messages": [
                {"role": "user", "content": [{"text": prompt}]}
            ],
            "inferenceConfig": {
                "maxTokens": 150,
                "temperature": 0.0
            }
        }
        
        response = self.bedrock_client.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        text = response_body['output']['message']['content'][0]['text']
        
        try:
            # Extract JSON from response
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
        
        # Default classification
        return {
            "type": "factual",
            "needs_tables": False,
            "needs_visuals": False,
            "complexity": "moderate"
        }
