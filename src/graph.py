from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from src.retrieval import RetrievalEngine
from src.generation import GenerationEngine

# Define Graph State
class AgentState(TypedDict):
    question: str
    documents: List[str]
    answer: str

class RAGGraph:
    def __init__(self):
        self.retriever = RetrievalEngine()
        self.generator = GenerationEngine()
        self.workflow = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Define Nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("rerank", self.rerank)
        workflow.add_node("generate", self.generate)

        # Define Edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "rerank")
        workflow.add_edge("rerank", "generate")
        workflow.add_edge("generate", END)

        return workflow.compile()

    # Node Functions
    def retrieve(self, state: AgentState):
        """
        Retrieves documents based on the question.
        """
        print("---RETRIEVING---")
        question = state["question"]
        docs = self.retriever.query(question)
        return {"documents": [d.page_content for d in docs]}

    def rerank(self, state: AgentState):
        """
        Reranks retrieved documents. 
        Note: The actual method in retrieval.py returns docs, here we assume it processes text lists or objects.
        Adapted for the simplified state use.
        """
        print("---RERANKING---")
        # In a real app, pass objects. Here we just pass through for the mock.
        return {"documents": state["documents"]}

    def generate(self, state: AgentState):
        """
        Generates answer using LLM.
        """
        print("---GENERATING---")
        question = state["question"]
        docs_txt = "\n\n".join(state["documents"])
        answer = self.generator.generate_answer(docs_txt, question)
        return {"answer": answer}

    def run(self, question: str):
        """
        Main entry point to run the graph.
        """
        inputs = {"question": question}
        return self.workflow.invoke(inputs)

    def ingest_document(self, file_path: str):
        """
        Helper to ingest a doc via the retrieval engine.
        Using a simple mock ingestion for the graph wrapper.
        Real ingestion uses src.ingestion.DocumentIngester
        """
        from src.ingestion import DocumentIngester
        ingester = DocumentIngester()
        text = ingester.ingest(file_path)
        self.retriever.add_documents(text, metadata={"source": file_path})
        return "Ingestion Complete"
