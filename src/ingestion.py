import boto3
import time
import logging
from typing import List, Dict, Any
from pathlib import Path

from src.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextractProcessor:
    def __init__(self):
        self.client = boto3.client(
            'textract',
            region_name=config.AWS_REGION,
            aws_access_key_id=config.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=config.AWS_SECRET_ACCESS_KEY
        )

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Ingests a PDF/Image using Amazon Textract.
        Simulates async job for multi-page PDFs or sync for images.
        """
        logger.info(f"Starting Textract processing for: {file_path}")
        
        # NOTE: For this plausible code, we are mocking the actual bytes I/O for simplicity
        # In a real scenario, you'd upload to S3 first for multi-page PDFs.
        
        with open(file_path, 'rb') as doc:
            doc_bytes = doc.read()

        # Determining if we assume a single page byte stream for simplicity
        # or mock the S3 flow. We will stick to request_document_analysis for complex layout.
        
        try:
            # Using synchronous call for simplicity in this demo wrapper
            # For large PDFs, StartDocumentAnalysis (Async) is preferred.
            response = self.client.analyze_document(
                Document={'Bytes': doc_bytes},
                FeatureTypes=['TABLES', 'FORMS', 'LAYOUT']
            )
            
            return self._parse_textract_response(response)
        except Exception as e:
            logger.error(f"Textract failed: {e}")
            return {"text": "", "tables": [], "visual_elements": []}

    def _parse_textract_response(self, response: Dict) -> Dict[str, Any]:
        """
        Parses the JSON response from Textract into structured text, tables, and bounding boxes.
        """
        blocks = response.get('Blocks', [])
        
        lines = []
        tables = []
        visual_candidates = [] # Figures/Diagrams
        
        for block in blocks:
            if block['BlockType'] == 'LINE':
                lines.append(block['Text'])
            elif block['BlockType'] == 'TABLE':
                # Sophisticated table extraction logic would go here
                tables.append(block)
            elif block['BlockType'] in ['FIGURE', 'PICTURE']: # If supported by Layout
                visual_candidates.append(block)
                
        full_text = "\n".join(lines)
        
        return {
            "text": full_text,
            "raw_blocks": blocks,
            "tables_count": len(tables),
            "visuals_count": len(visual_candidates)
        }

class DocumentIngester:
    def __init__(self):
        self.textract = TextractProcessor()
        
    def ingest(self, file_path: str):
        """
        Main entry point:
        1. OCR with Textract
        2. Split text
        3. Prepare for Vector Store
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{file_path} not found.")
            
        extraction_result = self.textract.process_file(file_path)
        
        # Here we would integrate the VLM logic:
        # If extraction_result['visuals_count'] > 0:
        #    crop images -> send to GPT-4o -> get description -> append to text
        
        return extraction_result["text"]
