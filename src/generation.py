from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.config import config

class GenerationEngine:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0,
            openai_api_key=config.OPENAI_API_KEY
        )
        
        self.rag_prompt = ChatPromptTemplate.from_template(
            """You are an advanced assistant for analyzing complex technical documents.
            Use the following context to answer the user's question. 
            If the context contains tables or visual descriptions, integrate them into your answer.
            
            Context:
            {context}
            
            Question: 
            {question}
            
            Answer:"""
        )
        
        self.chain = self.rag_prompt | self.llm | StrOutputParser()

    def generate_answer(self, context: str, question: str) -> str:
        return self.chain.invoke({"context": context, "question": question})

    def analyze_image(self, image_b64: str, prompt: str) -> str:
        """
        Multimodal visual reasoning using GPT-4o.
        """
        # Construct message payload for vision model
        message = self.llm.invoke(
            [
                ("human", [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ])
            ]
        )
        return message.content
