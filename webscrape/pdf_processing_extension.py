"""
PDF Processing Extension for Indian Legal Data Scraper

This script extends the main scraper to handle PDF files from government websites.
It processes PDFs containing legal documents and extracts structured text.
"""

import os
import re
import json
import csv
import logging
import requests
from io import BytesIO
import pandas as pd

# For PDF processing
try:
    import PyPDF2
    PDF_LIBRARY = "PyPDF2"
except ImportError:
    try:
        import pdfplumber
        PDF_LIBRARY = "pdfplumber"
    except ImportError:
        PDF_LIBRARY = None

# Set up logging - uses the same configuration as the main script
logger = logging.getLogger("pdf_processor")

class PDFProcessor:
    def __init__(self, pdf_sources_file='data/csv/legislative_dept_laws.csv', output_dir='data'):
        """Initialize the PDF processor with source file and output directory"""
        self.pdf_sources_file = pdf_sources_file
        self.output_dir = output_dir
        self.processed_pdfs = []
        
        # Check if PDF processing is available
        if PDF_LIBRARY is None:
            logger.warning("No PDF processing library found. Install PyPDF2 or pdfplumber.")
            print("Warning: To process PDFs, install PyPDF2 or pdfplumber with:")
            print("  pip install PyPDF2")
            print("  or")
            print("  pip install pdfplumber")
    
    def process_all_pdfs(self):
        """Process all PDFs listed in the source file"""
        if PDF_LIBRARY is None:
            logger.error("Cannot process PDFs: No PDF library installed")
            return
        
        logger.info(f"Processing PDFs using {PDF_LIBRARY}")
        
        try:
            # Read the CSV with PDF sources
            df = pd.read_csv(self.pdf_sources_file)
            
            for index, row in df.iterrows():
                if row.get('requires_pdf_processing', False) or '.pdf' in row.get('url', ''):
                    self.process_single_pdf(row['title'], row['url'], row['source'])
            
            # Save the processed data
            self.save_data()
            
            logger.info(f"Processed {len(self.processed_pdfs)} PDF files")
            
        except Exception as e:
            logger.error(f"Error processing PDFs: {str(e)}", exc_info=True)
    
    def process_single_pdf(self, title, url, source):
        """Process a single PDF file"""
        logger.info(f"Processing PDF: {title} from {url}")
        
        try:
            # Download the PDF
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Process the PDF content
            pdf_content = BytesIO(response.content)
            extracted_text = self._extract_text_from_pdf(pdf_content)
            
            # Parse the text into sections
            sections = self._parse_sections(extracted_text)
            
            # Create a structured record
            pdf_data = {
                'source': source,
                'type': 'Act' if 'act' in title.lower() else 'Constitution',
                'title': title,
                'url': url,
                'content': extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                'sections': sections,
                'full_text_length': len(extracted_text)
            }
            
            # Save the raw text
            sanitized_title = re.sub(r'[^\w\s-]', '', title).strip().replace(' ', '_')
            with open(f"{self.output_dir}/raw/{sanitized_title}.txt", 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            self.processed_pdfs.append(pdf_data)
            logger.info(f"Successfully processed PDF: {title}")
            
        except Exception as e:
            logger.error(f"Error processing PDF {title}: {str(e)}", exc_info=True)
    
    def _extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file using available library"""
        text = ""
        
        if PDF_LIBRARY == "PyPDF2":
            reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text() + "\n\n"
                
        elif PDF_LIBRARY == "pdfplumber":
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n\n"
        
        return text.strip()
    
    def _parse_sections(self, text):
        """Parse the extracted text into sections"""
        sections = []
        
        # Pattern for sections in legal documents
        # This is a simplified pattern and might need adjustment based on actual formats
        section_pattern = re.compile(r'(?:Section|SECTION)\s+(\d+)[.\s:]+(.+?)(?=(?:Section|SECTION)\s+\d+|$)', re.DOTALL)
        
        # Find all sections
        for match in section_pattern.finditer(text):
            section_num = match.group(1)
            section_content = match.group(2).strip()
            
            sections.append({
                'number': section_num,
                'title': '',  # Title might be part of the content
                'content': section_content
            })
        
        # If no sections found, try alternative patterns
        if not sections:
            # Try to find articles for Constitution
            article_pattern = re.compile(r'(?:Article|ARTICLE)\s+(\d+)[.\s:]+(.+?)(?=(?:Article|ARTICLE)\s+\d+|$)', re.DOTALL)
            
            for match in article_pattern.finditer(text):
                article_num = match.group(1)
                article_content = match.group(2).strip()
                
                sections.append({
                    'number': article_num,
                    'title': '',
                    'content': article_content
                })
        
        # If still no sections found, split by common patterns in legal documents
        if not sections:
            # Try to split by numbered paragraphs (1., 2., etc.)
            number_pattern = re.compile(r'(\d+)\.\s+(.+?)(?=\d+\.\s+|$)', re.DOTALL)
            
            for match in number_pattern.finditer(text):
                para_num = match.group(1)
                para_content = match.group(2).strip()
                
                sections.append({
                    'number': para_num,
                    'title': '',
                    'content': para_content
                })
        
        return sections
    
    def save_data(self):
        """Save the processed PDF data to files"""
        if not self.processed_pdfs:
            logger.warning("No processed PDFs to save")
            return
        
        # Save as JSON
        with open(f'{self.output_dir}/json/processed_pdfs.json', 'w', encoding='utf-8') as f:
            json.dump(self.processed_pdfs, f, ensure_ascii=False, indent=2)
        
        # Save main CSV
        with open(f'{self.output_dir}/csv/processed_pdfs.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'type', 'title', 'url', 'full_text_length'])
            
            for pdf in self.processed_pdfs:
                writer.writerow([
                    pdf['source'],
                    pdf['type'],
                    pdf['title'], 
                    pdf['url'],
                    pdf['full_text_length']
                ])
        
        # Save sections as a separate CSV
        sections_data = []
        
        for pdf in self.processed_pdfs:
            if 'sections' in pdf:
                for section in pdf['sections']:
                    sections_data.append({
                        'law_title': pdf['title'],
                        'section_number': section['number'],
                        'section_title': section['title'],
                        'section_content': section['content'][:500] + "..." if len(section['content']) > 500 else section['content']
                    })
        
        if sections_data:
            pd.DataFrame(sections_data).to_csv(f'{self.output_dir}/csv/pdf_sections.csv', index=False)
        
        logger.info(f"Saved data for {len(self.processed_pdfs)} processed PDFs")


# If run directly
if __name__ == "__main__":
    # Set up directories
    os.makedirs('data/csv', exist_ok=True)
    os.makedirs('data/json', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("pdf_processor.log"),
            logging.StreamHandler()
        ]
    )
    
    # Process PDFs
    processor = PDFProcessor()
    processor.process_all_pdfs()