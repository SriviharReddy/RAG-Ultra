"""
Document Ingestion Module with Amazon Textract OCR and Vision Processing
Handles PDF ingestion, OCR extraction, table detection, and formula recognition
"""

import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import boto3
from PIL import Image
from langchain_core.documents import Document
from pdf2image import convert_from_path, convert_from_bytes

from src.config import config


@dataclass
class ExtractedContent:
    """Represents extracted content from a document page"""
    text: str
    tables: List[Dict[str, Any]]
    images: List[str]  # Base64 encoded images
    page_number: int
    confidence: float
    content_type: str  # 'text', 'table', 'image', 'formula'


class TextractProcessor:
    """
    Amazon Textract processor for OCR, table extraction, and document analysis
    """
    
    def __init__(self):
        self.textract_client = boto3.client(
            'textract',
            region_name=config.aws.region
        )
    
    def extract_text_from_bytes(self, document_bytes: bytes) -> Dict[str, Any]:
        """
        Extract text from document bytes using Amazon Textract
        """
        response = self.textract_client.analyze_document(
            Document={'Bytes': document_bytes},
            FeatureTypes=['TABLES', 'FORMS', 'LAYOUT']
        )
        return self._parse_textract_response(response)
    
    def extract_from_s3(self, bucket: str, key: str) -> Dict[str, Any]:
        """
        Extract text from document in S3 using async job for large documents
        """
        response = self.textract_client.start_document_analysis(
            DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': key}},
            FeatureTypes=['TABLES', 'FORMS', 'LAYOUT']
        )
        job_id = response['JobId']
        return self._wait_for_job(job_id)
    
    def _wait_for_job(self, job_id: str) -> Dict[str, Any]:
        """Wait for async Textract job to complete"""
        import time
        while True:
            response = self.textract_client.get_document_analysis(JobId=job_id)
            status = response['JobStatus']
            if status == 'SUCCEEDED':
                return self._parse_textract_response(response)
            elif status == 'FAILED':
                raise Exception(f"Textract job failed: {response.get('StatusMessage')}")
            time.sleep(2)
    
    def _parse_textract_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse Textract response into structured content
        """
        blocks = response.get('Blocks', [])
        
        text_content = []
        tables = []
        forms = []
        
        block_map = {block['Id']: block for block in blocks}
        
        for block in blocks:
            block_type = block.get('BlockType')
            
            if block_type == 'LINE':
                text_content.append({
                    'text': block.get('Text', ''),
                    'confidence': block.get('Confidence', 0),
                    'geometry': block.get('Geometry', {})
                })
            
            elif block_type == 'TABLE':
                table = self._extract_table(block, block_map)
                tables.append(table)
            
            elif block_type == 'KEY_VALUE_SET':
                if 'KEY' in block.get('EntityTypes', []):
                    form_item = self._extract_form_item(block, block_map)
                    if form_item:
                        forms.append(form_item)
        
        return {
            'text': '\n'.join([t['text'] for t in text_content]),
            'text_blocks': text_content,
            'tables': tables,
            'forms': forms,
            'raw_blocks': blocks
        }
    
    def _extract_table(self, table_block: Dict, block_map: Dict) -> Dict[str, Any]:
        """Extract table structure from Textract blocks"""
        cells = []
        
        if 'Relationships' in table_block:
            for relationship in table_block['Relationships']:
                if relationship['Type'] == 'CHILD':
                    for cell_id in relationship['Ids']:
                        cell_block = block_map.get(cell_id, {})
                        if cell_block.get('BlockType') == 'CELL':
                            cell_text = self._get_text_from_children(cell_block, block_map)
                            cells.append({
                                'row': cell_block.get('RowIndex', 0),
                                'col': cell_block.get('ColumnIndex', 0),
                                'text': cell_text,
                                'row_span': cell_block.get('RowSpan', 1),
                                'col_span': cell_block.get('ColumnSpan', 1)
                            })
        
        # Convert to 2D table
        if cells:
            max_row = max(c['row'] for c in cells)
            max_col = max(c['col'] for c in cells)
            table_data = [['' for _ in range(max_col)] for _ in range(max_row)]
            
            for cell in cells:
                row_idx = cell['row'] - 1
                col_idx = cell['col'] - 1
                if 0 <= row_idx < max_row and 0 <= col_idx < max_col:
                    table_data[row_idx][col_idx] = cell['text']
            
            return {'data': table_data, 'raw_cells': cells}
        
        return {'data': [], 'raw_cells': []}
    
    def _extract_form_item(self, key_block: Dict, block_map: Dict) -> Optional[Dict[str, str]]:
        """Extract form key-value pair"""
        key_text = self._get_text_from_children(key_block, block_map)
        value_text = ''
        
        if 'Relationships' in key_block:
            for relationship in key_block['Relationships']:
                if relationship['Type'] == 'VALUE':
                    for value_id in relationship['Ids']:
                        value_block = block_map.get(value_id, {})
                        value_text = self._get_text_from_children(value_block, block_map)
        
        if key_text:
            return {'key': key_text, 'value': value_text}
        return None
    
    def _get_text_from_children(self, block: Dict, block_map: Dict) -> str:
        """Get text content from child blocks"""
        text_parts = []
        
        if 'Relationships' in block:
            for relationship in block['Relationships']:
                if relationship['Type'] == 'CHILD':
                    for child_id in relationship['Ids']:
                        child_block = block_map.get(child_id, {})
                        if child_block.get('BlockType') == 'WORD':
                            text_parts.append(child_block.get('Text', ''))
        
        return ' '.join(text_parts)


class VisionProcessor:
    """
    Vision processor for image understanding using Bedrock vision models
    """
    
    def __init__(self):
        self.bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=config.aws.region
        )
    
    def describe_image(self, image: Image.Image, context: str = "") -> str:
        """
        Use vision model to describe image content, including diagrams, charts, formulas
        """
        # Convert image to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        image_base64 = base64.standard_b64encode(image_bytes).decode('utf-8')
        
        prompt = f"""Analyze this image in detail. 
If it contains:
- Text: Extract and transcribe all visible text
- Tables: Describe the structure and content
- Charts/Graphs: Describe the data and trends shown
- Diagrams: Explain the components and relationships
- Mathematical formulas: Convert to LaTeX notation
- Code: Transcribe the code exactly

Context: {context if context else 'Document page analysis'}

Provide a comprehensive description that captures all information."""

        # Using Nova Lite for vision tasks
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": "png",
                                "source": {"bytes": image_base64}
                            }
                        },
                        {"text": prompt}
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 2048,
                "temperature": 0.1
            }
        }
        
        import json
        response = self.bedrock_client.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['output']['message']['content'][0]['text']
    
    def extract_formula(self, image: Image.Image) -> str:
        """
        Extract mathematical formulas from image and convert to LaTeX
        """
        prompt = """This image contains a mathematical formula or equation.
Please convert it to proper LaTeX notation.
Only output the LaTeX code, wrapped in $$ delimiters for display math or $ for inline math.
If there are multiple formulas, separate them with line breaks."""

        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        image_base64 = base64.standard_b64encode(image_bytes).decode('utf-8')
        
        import json
        request_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": "png",
                                "source": {"bytes": image_base64}
                            }
                        },
                        {"text": prompt}
                    ]
                }
            ],
            "inferenceConfig": {
                "maxTokens": 1024,
                "temperature": 0.0
            }
        }
        
        response = self.bedrock_client.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps(request_body),
            contentType="application/json"
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['output']['message']['content'][0]['text']


class DocumentIngester:
    """
    Main document ingestion pipeline combining OCR and vision processing
    """
    
    def __init__(self):
        self.textract = TextractProcessor()
        self.vision = VisionProcessor()
    
    def ingest_pdf(self, pdf_path: str) -> List[Document]:
        """
        Ingest a PDF document with full multimodal processing
        """
        pdf_path = Path(pdf_path)
        documents = []
        
        # Convert PDF to images for vision processing
        images = convert_from_path(str(pdf_path), dpi=300)
        
        # Read PDF bytes for Textract
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        
        # Process each page
        for page_num, image in enumerate(images, 1):
            # Convert single page to bytes for Textract
            page_buffer = io.BytesIO()
            image.save(page_buffer, format='PNG')
            page_bytes = page_buffer.getvalue()
            
            # OCR extraction with Textract
            textract_result = self.textract.extract_text_from_bytes(page_bytes)
            
            # Vision analysis for complex content
            vision_description = self.vision.describe_image(
                image, 
                context=f"Page {page_num} of document: {pdf_path.name}"
            )
            
            # Create parent document (full page content)
            parent_content = self._create_parent_content(
                textract_result, 
                vision_description, 
                page_num
            )
            
            parent_doc = Document(
                page_content=parent_content,
                metadata={
                    'source': str(pdf_path),
                    'page': page_num,
                    'total_pages': len(images),
                    'doc_type': 'parent',
                    'has_tables': len(textract_result.get('tables', [])) > 0,
                    'has_forms': len(textract_result.get('forms', [])) > 0
                }
            )
            documents.append(parent_doc)
            
            # Create child documents for tables
            for idx, table in enumerate(textract_result.get('tables', [])):
                table_content = self._format_table(table)
                if table_content:
                    table_doc = Document(
                        page_content=table_content,
                        metadata={
                            'source': str(pdf_path),
                            'page': page_num,
                            'doc_type': 'table',
                            'table_index': idx,
                            'parent_page': page_num
                        }
                    )
                    documents.append(table_doc)
        
        return documents
    
    def ingest_pdf_bytes(self, pdf_bytes: bytes, filename: str) -> List[Document]:
        """
        Ingest PDF from bytes (for uploaded files)
        """
        documents = []
        
        # Convert bytes to images
        images = convert_from_bytes(pdf_bytes, dpi=300)
        
        for page_num, image in enumerate(images, 1):
            # Convert to bytes for processing
            page_buffer = io.BytesIO()
            image.save(page_buffer, format='PNG')
            page_bytes = page_buffer.getvalue()
            
            # OCR extraction
            textract_result = self.textract.extract_text_from_bytes(page_bytes)
            
            # Vision analysis
            vision_description = self.vision.describe_image(
                image,
                context=f"Page {page_num} of document: {filename}"
            )
            
            # Create documents
            parent_content = self._create_parent_content(
                textract_result,
                vision_description,
                page_num
            )
            
            parent_doc = Document(
                page_content=parent_content,
                metadata={
                    'source': filename,
                    'page': page_num,
                    'total_pages': len(images),
                    'doc_type': 'parent',
                    'has_tables': len(textract_result.get('tables', [])) > 0
                }
            )
            documents.append(parent_doc)
            
            # Table documents
            for idx, table in enumerate(textract_result.get('tables', [])):
                table_content = self._format_table(table)
                if table_content:
                    table_doc = Document(
                        page_content=table_content,
                        metadata={
                            'source': filename,
                            'page': page_num,
                            'doc_type': 'table',
                            'table_index': idx
                        }
                    )
                    documents.append(table_doc)
        
        return documents
    
    def _create_parent_content(
        self, 
        textract_result: Dict[str, Any], 
        vision_description: str,
        page_num: int
    ) -> str:
        """
        Create comprehensive parent document content
        """
        parts = [f"=== Page {page_num} ===\n"]
        
        # Add OCR text
        if textract_result.get('text'):
            parts.append("### Extracted Text ###\n")
            parts.append(textract_result['text'])
            parts.append("\n")
        
        # Add form data
        if textract_result.get('forms'):
            parts.append("\n### Form Fields ###\n")
            for form in textract_result['forms']:
                parts.append(f"- {form['key']}: {form['value']}\n")
        
        # Add vision analysis
        if vision_description:
            parts.append("\n### Visual Analysis ###\n")
            parts.append(vision_description)
            parts.append("\n")
        
        return ''.join(parts)
    
    def _format_table(self, table: Dict[str, Any]) -> str:
        """
        Format table data as markdown
        """
        data = table.get('data', [])
        if not data:
            return ""
        
        lines = ["### Table ###\n"]
        
        # Header row
        if data:
            lines.append("| " + " | ".join(str(cell) for cell in data[0]) + " |")
            lines.append("| " + " | ".join(["---"] * len(data[0])) + " |")
        
        # Data rows
        for row in data[1:]:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        
        return '\n'.join(lines)
