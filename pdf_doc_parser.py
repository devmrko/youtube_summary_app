from pathlib import Path
import logging
from datetime import datetime
from typing import List, Dict
import re
import pdfplumber
import os
import tabula
from tabulate import tabulate
import pdf2image
import pandas as pd
import pytesseract
import json
import openpyxl
import numpy as np
from PIL import Image
import subprocess
import sys
import deepl
import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    TextContent, Message, GenericChatRequest,
    ChatDetails, OnDemandServingMode
)
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyarleil5jr7k2rykljkhapnvhrqvzx4cwuvtfedlfxet4q"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFDocumentParser:

    def __init__(self, output_dir: str = "parsed_pdfs"):
        """Initialize the Enhanced PDF Table Parser"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Check if Tesseract is installed
        self.tesseract_available = self._check_tesseract()
        if not self.tesseract_available:
            self.logger.warning("Tesseract OCR is not installed or not in PATH.")
            self.logger.warning("OCR functionality will be limited.")
            self.logger.warning("See README file for Tesseract installation instructions.")

    def _check_tesseract(self) -> bool:
        """Check if Tesseract OCR is installed and accessible"""
        try:
            # Try to run tesseract --version
            subprocess.run(
                ["tesseract", "--version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                check=False,
                timeout=5
            )
            return True
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            # If any error occurs, tesseract is not properly installed
            return False
            
    def _identify_subjects(self, text: str) -> List[Dict]:
        """
        Identify subject patterns like "1 Introduction", "2 Background" in text content
        and separate them into structured sections.
        
        Returns:
            List of dictionaries with 'title' and 'content' keys
        """
        # Define patterns to identify subject headings - focus on simple number patterns
        subject_patterns = [
            # Pattern for simple numbered headings: "1 Introduction", "2 Background", etc.
            r'(?:^|\n)(\d+)[\.\s]+([A-Z][^\n]+)',
            
            # Pattern for decimal numbered headings: "1.1 Method", "2.3 Results", etc.
            r'(?:^|\n)(\d+\.\d+)[\.\s]+([A-Z][^\n]+)',
            
            # Pattern for multi-level numbering: "1.2.3 Analysis", etc.
            r'(?:^|\n)(\d+(?:\.\d+){1,2})[\.\s]+([A-Z][^\n]+)',
            
            # Pattern for simple numbers at beginning of line with title case
            r'(?:^|\n)(\d+)[\.\s]+([A-Z][a-z]+[^\n]*)',
            
            # Pattern for Roman numerals: "I. Introduction", "II. Background", etc.
            r'(?:^|\n)((?:I|V|X|L|C|D|M)+)[\.\s]+([A-Z][^\n]+)'
        ]
        
        # Find all potential subject matches
        all_matches = []
        for pattern in subject_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                # Get the match position and the subject number
                start_pos = match.start()
                subject_num = match.group(1)
                subject_heading = match.group(2).strip() if len(match.groups()) > 1 else ""
                subject_title = f"{subject_num} {subject_heading}"
                
                # Don't add duplicates
                if not any(m['start'] == start_pos for m in all_matches):
                    all_matches.append({
                        'start': start_pos,
                        'title': subject_title,
                        'number': subject_num,
                        'heading': subject_heading
                    })
        
        # Sort matches by their position in the text
        all_matches.sort(key=lambda x: x['start'])
        
        # If no matches found, return the whole text as one section
        if not all_matches:
            return [{'title': 'Main Content', 'content': text.strip()}]
        
        # Create sections based on the matches
        sections = []
        for i, match in enumerate(all_matches):
            start = match['start']
            # End is either the start of the next match or the end of the text
            end = all_matches[i+1]['start'] if i < len(all_matches) - 1 else len(text)
            
            # Extract the content between this match and the next one
            content = text[start:end].strip()
            
            # Skip empty sections
            if not content:
                continue
                
            sections.append({
                'title': match['title'],
                'number': match['number'],
                'heading': match['heading'],
                'content': content
            })
            
        self.logger.info(f"Identified {len(sections)} numbered sections in the document")
        return sections

    def extract_text_content(self, pdf_path: str) -> Dict:
        """Extract text content and structure from PDF while preserving original spacing and special characters"""
        try:
            content = {
                'title': '',
                'abstract': '',
                'sections': [],
                'subjects': [],  # New field for identified subjects
                'references': [],
                'raw_text': '',
                'pages': []
            }
            
            # First try with pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                logger.info("Extracting text content from PDF while preserving structure")
                full_text = ""
                for page_num, page in enumerate(pdf.pages, 1):
                    # Get page dimensions
                    width = page.width
                    height = page.height
                    
                    # Extract text with exact positioning
                    words = page.extract_words(
                        x_tolerance=1,  # Minimal x tolerance to preserve spacing
                        y_tolerance=1,  # Minimal y tolerance to preserve line spacing
                        keep_blank_chars=True,  # Keep spaces
                    )
                    
                    # Sort words by vertical position first, then horizontal
                    words.sort(key=lambda w: (round(w['top']), w['x0']))
                    
                    # Group words by line (based on vertical position)
                    current_line_top = -1
                    current_line = []
                    lines = []
                    
                    for word in words:
                        # Process special characters in the word
                        word_text = word['text']
                        
                        # If this is a new line
                        if current_line_top == -1 or abs(word['top'] - current_line_top) > 3:
                            if current_line:
                                lines.append(current_line)
                            current_line = []
                            current_line_top = word['top']
                        
                        # Add word to current line with preserved spacing
                        if current_line:
                            # Calculate space between words
                            last_word = current_line[-1]
                            space_width = word['x0'] - last_word['x1']
                            spaces = ' ' * max(1, round(space_width / 5))  # Approximate space width
                            current_line.append({'text': spaces + word_text, 'x0': word['x0'], 'x1': word['x1']})
                        else:
                            # First word in line - preserve left margin
                            left_margin = ' ' * max(0, round(word['x0'] / 5))
                            current_line.append({'text': left_margin + word_text, 'x0': word['x0'], 'x1': word['x1']})
                    
                    # Add last line
                    if current_line:
                        lines.append(current_line)
                    
                    # Convert lines to text with preserved spacing
                    page_text = '\n'.join(''.join(word['text'] for word in line) for line in lines)
                    
                    # Add extra newline between pages
                    if page_text:
                        content['pages'].append({
                            'page_number': page_num,
                            'text': page_text
                        })
                        full_text += page_text + '\n\n'
            
            # Store complete raw text
            content['raw_text'] = full_text
            
            # Process document structure while preserving spacing
            if full_text:
                lines = full_text.split('\n')
                
                # Find title (first non-empty line that's not metadata)
                for line in lines:
                    if line.strip() and not line.strip().lower().startswith(('doi:', 'http', 'www')):
                        content['title'] = line
                        break
                
                # Find abstract with preserved spacing
                abstract_patterns = [
                    r'(Abstract\s*[:\-—]*\s*.*?)(?=\n\s*(?:1\.?\s+)?Introduction|\Z)',
                    r'(ABSTRACT\s*[:\-—]*\s*.*?)(?=\n\s*(?:1\.?\s+)?Introduction|\Z)',
                    r'(Abstract\s*[:\-—]*\s*.*?)(?=\n\s*(?:\d+\.?\s+)?[A-Z][a-z]+\s*\n)',
                ]
                
                for pattern in abstract_patterns:
                    abstract_match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
                    if abstract_match:
                        content['abstract'] = abstract_match.group(1)
                        break
                
                # Identify subject patterns in the text
                identified_subjects = self._identify_subjects(full_text)
                content['subjects'] = identified_subjects
                
                # Find sections with preserved spacing (original section detection)
                section_patterns = [
                    r'(?:\n|\A)((?:\d+\.?\s+)?[A-Z][^\n]+)(?:\n|\Z)',  # Numbered sections
                    r'(?:\n|\A)([A-Z][A-Z\s]+[A-Z])(?:\n|\Z)',         # ALL CAPS sections
                    r'(?:\n|\A)([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s*)(?:\n|\Z)'  # Title Case sections
                ]
                
                current_section = None
                last_position = 0
                
                for pattern in section_patterns:
                    for section_match in re.finditer(pattern, full_text):
                        section_title = section_match.group(1)
                        section_start = section_match.end()
                        
                        if section_start < last_position:
                            continue
                        
                        if current_section is not None:
                            section_content = full_text[current_section['start']:section_match.start()]
                            if section_content:
                                content['sections'].append({
                                    'title': current_section['title'],
                                    'content': section_content
                                })
                        
                        current_section = {'title': section_title, 'start': section_start}
                        last_position = section_start
                
                # Get last section
                if current_section is not None:
                    section_content = full_text[current_section['start']:]
                    if section_content:
                        content['sections'].append({
                            'title': current_section['title'],
                            'content': section_content
                        })
                
                # Find references with preserved spacing
                reference_patterns = [
                    r'(References\s*.*?)(?=\n\s*(?:\d+\.?\s+)?[A-Z]|\Z)',
                    r'(REFERENCES\s*.*?)(?=\n\s*(?:\d+\.?\s+)?[A-Z]|\Z)',
                    r'(Bibliography\s*.*?)(?=\n\s*(?:\d+\.?\s+)?[A-Z]|\Z)',
                ]
                
                for pattern in reference_patterns:
                    references_match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
                    if references_match:
                        references_text = references_match.group(1)
                        # Split references while preserving indentation
                        references = re.split(r'\n(?=\[\d+\]|\d+\.|\[\d+\]|\(\d+\))', references_text)
                        content['references'] = [ref for ref in references if ref.strip()]
                        break
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error extracting text content: {str(e)}")
            return None
        
    def extract_tables(self, pdf_path: str, page_number: int = None) -> List[Dict]:
        """Extract tables using Tabula's detection with OCR enhancement"""
        tables = []

        try:
            print(f"# Extract tables using Tabula's detection with OCR enhancement")
            # Try to find poppler path
            poppler_path = None
            possible_paths = [
                r"C:\Program Files\poppler\Library\bin",
                r"C:\Program Files (x86)\poppler\Library\bin",
                r"C:\poppler\Library\bin",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "poppler", "Library", "bin")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    poppler_path = path
                    break
            
            if not poppler_path:
                print("\nPoppler not found. Please install Poppler:")
                print("1. Download from: https://github.com/oschwartz10612/poppler-windows/releases/")
                print("2. Extract to C:\\Program Files\\poppler")
                print("3. Add C:\\Program Files\\poppler\\Library\\bin to PATH")
                raise Exception("Poppler not found in system PATH")
            
            # Determine which pages to process
            pages_param = 'all' if page_number is None else page_number
            self.logger.info(f"Processing page(s): {pages_param}")
            
            # Convert only the specific page if page_number is provided
            if page_number is not None:
                # pdf2image uses 1-indexed page numbers
                images = pdf2image.convert_from_path(
                    pdf_path, 
                    poppler_path=poppler_path,
                    first_page=page_number,
                    last_page=page_number
                )
                self.logger.info(f"Converted page {page_number} to image")
            else:
                return None

            try:
                # Extract tables using Tabula for structure
                dfs = tabula.read_pdf(
                    pdf_path,
                    pages=pages_param,
                    multiple_tables=True,
                    guess=True,
                    encoding='utf-8',
                    java_options=['-Dfile.encoding=UTF8'],
                    pandas_options={
                        'header': None,
                        'encoding': 'utf-8'
                    }
                )
                
                # Handle case where dfs is not a list
                if not isinstance(dfs, list):
                    dfs = [dfs]
                
                # Handle empty results
                if not dfs or len(dfs) == 0:
                    self.logger.info(f"No tables found on page {page_number}")
                    return []

                # Process each dataframe (table)
                for i, df in enumerate(dfs):
                    if df is None or df.empty:
                        self.logger.info(f"Empty table found, skipping")
                        continue
                        
                    self.logger.info(f"Processing table {i+1}")
                    
                    # For single page conversion, there's only one image at index 0
                    image_idx = 0 if page_number is not None else i
                    # Make sure we don't exceed the images array length
                    if image_idx >= len(images):
                        self.logger.warning(f"Image index {image_idx} exceeds images array length {len(images)}")
                        continue
                        
                    # Get table bounds from Tabula
                    current_page = page_number if page_number is not None else i + 1
                    table_areas = tabula.read_pdf(
                        pdf_path,
                        pages=current_page,
                        multiple_tables=True,
                        guess=True,
                        encoding='utf-8',
                        output_format='json'
                    )
                    
                    # Process each table area
                    for table_area in table_areas:
                        # Use appropriate image based on single/multi page conversion
                        df_enhanced = self._enhance_table_with_ocr(df, images[image_idx], table_area)
                        processed_table = self._process_table(df_enhanced, current_page)
                        if processed_table:
                            tables.append(processed_table)
                            self.logger.info(f"Extracted table from page {current_page}")
                    
                return tables
            except Exception as e:
                self.logger.error(f"Error extracting tables: {str(e)}")
                return None

        except Exception as e:
            self.logger.error(f"Error extracting tables: {str(e)}")
            raise
    
    def _enhance_table_with_ocr(self, df: pd.DataFrame, page_image: Image.Image, table_area: Dict) -> pd.DataFrame:
        """Enhance table data with OCR processing"""
        try:
            print(f"# Enhance table data with OCR processing")
            # Skip OCR processing if Tesseract is not available
            if not self.tesseract_available:
                self.logger.info("Skipping OCR enhancement because Tesseract is not available")
                return df
                
            # Create a copy of the DataFrame
            df_enhanced = df.copy()
            
            # Get table coordinates
            top = table_area['top']
            left = table_area['left']
            bottom = table_area['bottom']
            right = table_area['right']
            
            # Crop table area from page image
            table_image = page_image.crop((left, top, right, bottom))
            
            # Process each cell
            for i in range(len(df_enhanced)):
                for j in range(len(df_enhanced.columns)):
                    cell_value = str(df_enhanced.iloc[i, j]) if not pd.isna(df_enhanced.iloc[i, j]) else ""
                    
                    # Check if cell might contain scientific notation or special characters
                    if re.search(r'\d+\s*[·*]?\s*10?\d+', cell_value) or len(cell_value.strip()) == 0:
                        # Calculate cell position and crop
                        cell_height = (bottom - top) / len(df_enhanced)
                        cell_width = (right - left) / len(df_enhanced.columns)
                        
                        # Calculate cell boundaries
                        cell_top = top + (i * cell_height)
                        cell_left = left + (j * cell_width)
                        cell_right = cell_left + cell_width
                        cell_bottom = cell_top + cell_height
                        
                        # Ensure coordinates are within image bounds
                        img_width, img_height = page_image.size
                        cell_left = max(0, cell_left)
                        cell_top = max(0, cell_top)
                        cell_right = min(img_width, cell_right)
                        cell_bottom = min(img_height, cell_bottom)
                        
                        # Only process if we have a valid area
                        if cell_right > cell_left and cell_bottom > cell_top:
                            try:
                                cell_image = page_image.crop((
                                    cell_left, cell_top,
                                    cell_right, cell_bottom
                                ))
                                
                                # Process cell with OCR
                                ocr_text = self._process_cell_with_ocr(cell_image)
                                if ocr_text:
                                    df_enhanced.iloc[i, j] = ocr_text
                            except Exception as e:
                                self.logger.error(f"Error processing cell {i},{j}: {str(e)}")
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing table with OCR: {str(e)}")
            return df
        
    def _process_table(self, df: pd.DataFrame, page_num: int) -> Dict:
        """Process extracted table and improve its structure"""
        try:
            print(f"# Process extracted table and improve its structure")
            
            # Clean the DataFrame
            df = self._clean_dataframe(df)
            
            if df.empty:
                return None
            
            # Format table in different styles
            markdown = tabulate(df, headers='keys', tablefmt='pipe', showindex=False)
            html = tabulate(df, headers='keys', tablefmt='html', showindex=False)
            latex = tabulate(df, headers='keys', tablefmt='latex', showindex=False)
            
            return {
                'data': df,
                'markdown': markdown,
                'html': html,
                'latex': latex,
                'metadata': {
                    'page': page_num,
                    'rows': len(df),
                    'columns': len(df.columns)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing table: {str(e)}")
            return None

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and improve DataFrame structure"""
        try:
            print(f"# Clean and improve DataFrame structure")
            # Remove empty rows and columns
            df = df.replace(r'^\s*$', np.nan, regex=True)
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            
            # Clean cell values
            df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
            
            # Improve header detection
            if not df.empty and len(df) > 0:
                if self._is_header_row(df.iloc[0]):
                    new_header = df.iloc[0]
                    df = df[1:]
                    df.columns = new_header
            
            # Remove duplicate rows
            df = df.drop_duplicates()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning DataFrame: {str(e)}")
            return pd.DataFrame()

    def _is_header_row(self, row: pd.Series) -> bool:
        """Improved header row detection"""
        print(f"# Improved header row detection")
        if row.empty:
            return False
        
        # Check header characteristics
        non_empty_cells = row.astype(str).str.strip().str.len() > 0
        short_cells = row.astype(str).str.len() < 50
        non_numeric = ~row.astype(str).str.replace('.', '').str.isdigit()
        
        return (non_empty_cells & short_cells & non_numeric).mean() > 0.8

    def _process_cell_with_ocr(self, cell_image: Image.Image) -> str:
        """Process a cell image with OCR to extract text"""
        try:
            if not self.tesseract_available:
                # Return empty string if Tesseract is not available
                return ""
                
            # Configure Tesseract for better recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789.×^*·E() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-="'
            return pytesseract.image_to_string(cell_image, config=custom_config).strip()
        except Exception as e:
            self.logger.error(f"Error in OCR processing: {str(e)}")
            return ""

    def save_tables(self, tables: List[Dict]) -> None:
        """Save tables in multiple formats with improved organization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, table in enumerate(tables, 1):
            try:
                # Create directory for this table
                table_dir = self.output_dir / f"table_{i}_page{table['metadata']['page']}_{timestamp}"
                table_dir.mkdir(exist_ok=True)
                
                # Save as CSV with BOM for Excel compatibility
                csv_path = table_dir / "table.csv"
                table['data'].to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                # Save as Excel with formatting
                excel_path = table_dir / "table.xlsx"
                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    table['data'].to_excel(writer, index=False, sheet_name='Table')
                    self._format_excel(writer, table['data'])
                
                # Save as Markdown
                md_path = table_dir / "table.md"
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write(table['markdown'])
                
                # Save as HTML
                html_path = table_dir / "table.html"
                html_content = f"""
                <html>
                <head>
                    <meta charset="UTF-8">
                    <style>
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    </style>
                </head>
                <body>
                    {table['html']}
                </body>
                </html>
                """
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                # Save metadata
                meta_path = table_dir / "metadata.json"
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(table['metadata'], f, indent=4, ensure_ascii=False)
                
                self.logger.info(f"Saved table to {table_dir}")
                
            except Exception as e:
                self.logger.error(f"Error saving table {i}: {str(e)}")
                
    def _format_excel(self, writer: pd.ExcelWriter, df: pd.DataFrame) -> None:
        """Apply formatting to Excel output"""
        try:
            worksheet = writer.sheets['Table']
            
            # Format header
            for col in range(len(df.columns)):
                cell = worksheet.cell(row=1, column=col+1)
                cell.font = openpyxl.styles.Font(bold=True)
                cell.fill = openpyxl.styles.PatternFill(start_color='F2F2F2', end_color='F2F2F2', fill_type='solid')
        except Exception as e:
            self.logger.error(f"Error formatting Excel: {str(e)}")
            # Continue without formatting

def find_all_subtitles(text):
    # Common patterns for numbered headings
    patterns = [
        r'(?:^|\n)(\d+)[\.\s]+([A-Z][^\n]+)',  # "1 Introduction" or "1. Introduction"
        r'(?:^|\n)(\d+\.\d+)[\.\s]+([A-Z][^\n]+)',  # "1.1 Method" 
        r'(?:^|\n)(\d+(?:\.\d+){1,2})[\.\s]+([A-Z][^\n]+)'  # "1.2.3 Analysis"
    ]
    
    # Find all matches
    all_headings = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.MULTILINE):
            start_pos = match.start()
            number = match.group(1)
            heading = match.group(2).strip()
            
            all_headings.append({
                'start': start_pos,
                'number': number,
                'heading': heading,
                'full_title': f"{number} {heading}"
            })
    
    # Sort by position in text
    all_headings.sort(key=lambda x: x['start'])
    return all_headings

def split_by_subtitles(text, headings):
    sections = []
    
    for i, heading in enumerate(headings):
        start = heading['start']
        # End is either the start of the next heading or the end of the text
        end = headings[i+1]['start'] if i < len(headings) - 1 else len(text)
        
        # Get content for this section
        section_text = text[start:end].strip()
        
        sections.append({
            'title': heading['full_title'],
            'number': heading['number'],
            'heading': heading['heading'],
            'content': section_text
        })
    
    return sections

def normalize_sections(sections):
    normalized_text = ""
    
    for section in sections:
        # Standardize the heading format
        normalized_heading = f"{section['number']}. {section['heading']}"
        
        # Add to normalized text with consistent formatting
        normalized_text += f"\n\n{normalized_heading}\n"
        normalized_text += "=" * len(normalized_heading) + "\n\n"  # Underline for clarity
        
        # Get the content without the heading (which is the first line)
        content_lines = section['content'].split('\n')[1:]
        content = '\n'.join(content_lines)
        
        # Normalize paragraph spacing
        paragraphs = [p.strip() for p in content.split('\n\n')]
        paragraphs = [p for p in paragraphs if p]  # Remove empty paragraphs
        
        # Add normalized paragraphs
        normalized_text += '\n\n'.join(paragraphs)
    
    return normalized_text.strip()

def normalize_document_by_subtitles(text):
    # Find all subtitles/headings
    headings = find_all_subtitles(text)
    
    # No headings found
    if not headings:
        return text
    
    # Split content by headings
    sections = split_by_subtitles(text, headings)
    
    # Normalize each section
    normalized_text = normalize_sections(sections)
    
    return normalized_text

def split_by_known_subtitles(text: str, subtitle_list: List[str]) -> List[Dict]:
    """
    Split text based on a known list of subtitles.
    
    Args:
        text: The full document text
        subtitle_list: List of known subtitles in order (e.g., ["1 Introduction", "2 Background"])
    
    Returns:
        List of section dictionaries with title, number, heading, and content
    """
    sections = []
    logger.info(f"Splitting text by {len(subtitle_list)} known subtitles")
    
    # Find positions of all subtitles
    subtitle_positions = []
    for subtitle in subtitle_list:
        pos = text.find(subtitle)
        if pos != -1:
            subtitle_positions.append({
                'title': subtitle,
                'position': pos
            })
    
    # Sort by position
    subtitle_positions.sort(key=lambda x: x['position'])
    
    # Create sections from positions
    for i, subtitle_info in enumerate(subtitle_positions):
        # Get start of current section
        start_pos = subtitle_info['position']
        
        # Get title
        title = subtitle_info['title']
        
        # Find end of current section (start of next section)
        end_pos = len(text)  # Default to end of text
        if i < len(subtitle_positions) - 1:
            end_pos = subtitle_positions[i+1]['position']
        
        # Extract section content
        section_text = text[start_pos:end_pos].strip()
        
        # Split into heading line and content
        content_parts = section_text.split('\n', 1)
        heading_line = content_parts[0].strip()
        
        # Content is everything after the first line (or empty if there's only a heading)
        content_text = content_parts[1].strip() if len(content_parts) > 1 else ""
        
        # Extract number and heading
        number = ""
        heading = heading_line
        match = re.match(r'(\d+(?:\.\d+)*)\s+(.*)', heading_line)
        if match:
            number = match.group(1)
            heading = match.group(2)
        
        # Determine heading level
        level = 1
        if '.' in number:
            level = number.count('.') + 1
        
        sections.append({
            'title': heading_line,
            'number': number,
            'heading': heading,
            'level': level,
            'content': section_text,
            'text': content_text
        })
        
        logger.info(f"Extracted section: {heading_line} ({len(content_text)} chars)")
    
    # If no sections were found, return the whole text as one section
    if not sections:
        logger.warning("No sections found, returning whole text as one section")
        return [{'title': 'Document Content', 'number': '', 'heading': 'Document Content', 'content': text, 'text': text, 'level': 1}]
    
    return sections

def _get_oci_llm_response(prompt_text: str, text: str) -> str:
    """Generate summary using OCI Generative AI"""
    import traceback
    import uuid
    
    request_id = str(uuid.uuid4())[:8]  # Generate a unique ID for this request
    logger.info(f"[OCI-{request_id}] Starting OCI request")
    
    try:
        logger.info(f"[OCI-{request_id}] Creating text content")
        content = oci.generative_ai_inference.models.TextContent()
        content.text = f"{prompt_text}\n\n{text}"
        content.type = "TEXT"
        logger.info(f"[OCI-{request_id}] Text content created with length: {len(content.text)} chars")
        
        # Create message
        logger.info(f"[OCI-{request_id}] Creating message")
        message = oci.generative_ai_inference.models.Message()
        message.role = "USER"
        message.content = [content]
        
        # Create chat request
        logger.info(f"[OCI-{request_id}] Creating chat request")
        chat_request = oci.generative_ai_inference.models.GenericChatRequest()
        chat_request.api_format = oci.generative_ai_inference.models.BaseChatRequest.API_FORMAT_GENERIC
        chat_request.messages = [message]
        
        # Set model parameters
        model_params = {
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.7,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stop": None
        }
        logger.info(f"[OCI-{request_id}] Setting model parameters: {model_params}")
        
        for key, value in model_params.items():
            setattr(chat_request, key, value)

        # Create chat details
        logger.info(f"[OCI-{request_id}] Creating chat details with model ID: {MODEL_ID}")
        chat_details = oci.generative_ai_inference.models.ChatDetails()
        chat_details.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
            model_id=MODEL_ID
        )
        chat_details.chat_request = chat_request
        
        # Log accessing config
        logger.info(f"[OCI-{request_id}] Loading OCI config")
        config = oci.config.from_file()
        chat_details.compartment_id = config["tenancy"]
        logger.info(f"[OCI-{request_id}] Setting compartment ID: {chat_details.compartment_id[:8]}...")

        # Create client
        logger.info(f"[OCI-{request_id}] Creating GenerativeAiInferenceClient")
        llm_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=config,
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=3600  # Single timeout value in seconds (1 hour)
        )
        logger.info(f"[OCI-{request_id}] Client created successfully")

        # Make the actual API call
        logger.info(f"[OCI-{request_id}] Sending chat request")
        response = llm_client.chat(chat_details)
        logger.info(f"[OCI-{request_id}] Received response with status: {response.status}")
        
        # Extract the text response
        result_text = response.data.chat_response.choices[0].message.content[0].text
        logger.info(f"[OCI-{request_id}] Successfully extracted response text of length {len(result_text)} chars")
        
        return result_text
    
    except Exception as e:
        logger.error(f"[OCI-{request_id}] ERROR-2000: Error getting OCI LLM response: {str(e)}")
        # Get full traceback
        trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"[OCI-{request_id}] Traceback: {trace}")
        return f"ERROR-2000: Failed to generate response: {str(e)}"

def _get_ollama_llm_response(prompt_text: str, text: str) -> str:
    """Generate text using Ollama"""
    import traceback
    import json
    import sys
    import uuid

    request_id = str(uuid.uuid4())[:8]  # Generate a unique ID for this request
    logger.info(f"[OLLAMA-{request_id}] Starting Ollama request")
    
    try:
        logger.info(f"[OLLAMA-{request_id}] Prompt: {prompt_text[:50]}{'...' if len(prompt_text) > 50 else ''}")
        logger.info(f"[OLLAMA-{request_id}] Text length: {len(text)} chars")
        
        # Check if OLLAMA_URL is set
        ollama_url = os.getenv("OLLAMA_URL")
        if not ollama_url:
            logger.error(f"[OLLAMA-{request_id}] ERROR-1001: OLLAMA_URL environment variable not set")
            return f"ERROR-1001: OLLAMA_URL environment variable not set. Please set OLLAMA_URL to your Ollama server address (e.g. http://localhost:11434)"
        
        logger.info(f"[OLLAMA-{request_id}] Using Ollama URL: {ollama_url}")
        
        # Set up Ollama LLM
        logger.info(f"[OLLAMA-{request_id}] Creating OllamaLLM instance")
        llm = OllamaLLM(
            model="mistral",
            base_url=ollama_url
        )
        
        # Log the LLM configuration
        logger.info(f"[OLLAMA-{request_id}] LLM Config: model=mistral, base_url={ollama_url}")

        # Check if the Ollama server is responding
        try:
            # Direct debug output for raw model list
            logger.info(f"[OLLAMA-{request_id}] Testing Ollama connection...")
            from langchain_core.utils.logging import get_llm_request_id
            # Direct import for debug
            import requests
            resp = requests.get(f"{ollama_url}/api/tags")
            if resp.status_code != 200:
                logger.error(f"[OLLAMA-{request_id}] ERROR-1002: Ollama server not responding correctly. Status code: {resp.status_code}")
                logger.error(f"[OLLAMA-{request_id}] Response: {resp.text[:200]}")
                return f"ERROR-1002: Ollama server not responding correctly. Status code: {resp.status_code}"
            else:
                logger.info(f"[OLLAMA-{request_id}] Ollama connection successful. Available models: {resp.json()}")
        except Exception as conn_e:
            logger.error(f"[OLLAMA-{request_id}] ERROR-1003: Failed to connect to Ollama server: {str(conn_e)}")
            logger.error(f"[OLLAMA-{request_id}] Connection traceback: {''.join(traceback.format_exception(type(conn_e), conn_e, conn_e.__traceback__))}")
            return f"ERROR-1003: Failed to connect to Ollama server: {str(conn_e)}"
        
        # Create a proper prompt template
        logger.info(f"[OLLAMA-{request_id}] Creating prompt template")
        prompt_template = PromptTemplate(
            template="{prompt}\n\n{text}",
            input_variables=["prompt", "text"]
        )
        logger.info(f"[OLLAMA-{request_id}] Prompt template created with variables: {prompt_template.input_variables}")
        
        # Create a simple LLM chain
        from langchain.chains import LLMChain
        logger.info(f"[OLLAMA-{request_id}] Creating LLMChain")
        chain = LLMChain(
            llm=llm,
            prompt=prompt_template
        )
        logger.info(f"[OLLAMA-{request_id}] Chain input variables: {chain.input_keys}")
        logger.info(f"[OLLAMA-{request_id}] Chain output variables: {chain.output_keys}")
        
        # Execute the chain with the input variables
        logger.info(f"[OLLAMA-{request_id}] Invoking chain")
        input_data = {
            "prompt": prompt_text,
            "text": text
        }
        logger.info(f"[OLLAMA-{request_id}] Input keys: {list(input_data.keys())}")
        
        response = chain.invoke(input_data)
        logger.info(f"[OLLAMA-{request_id}] Chain invoked successfully")
        logger.info(f"[OLLAMA-{request_id}] Response keys: {list(response.keys())}")
        
        # Check if 'text' key exists in response
        if "text" not in response:
            logger.error(f"[OLLAMA-{request_id}] ERROR-1004: 'text' key not found in response. Available keys: {list(response.keys())}")
            logger.error(f"[OLLAMA-{request_id}] Full response: {json.dumps(response, default=str)[:500]}")
            
            # Try alternative keys
            if len(response) > 0:
                first_key = list(response.keys())[0]
                logger.info(f"[OLLAMA-{request_id}] Using alternative key: {first_key}")
                return response[first_key]
            else:
                return "ERROR-1004: No output text found in response"
        
        # Success path
        logger.info(f"[OLLAMA-{request_id}] Successfully generated text of length {len(response['text'])} chars")
        return response["text"]
    
    except Exception as e:
        logger.error(f"[OLLAMA-{request_id}] ERROR-1000: Error getting Ollama LLM response: {str(e)}")
        # Get full traceback
        trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"[OLLAMA-{request_id}] Traceback: {trace}")
        
        # Log the current state 
        logger.error(f"[OLLAMA-{request_id}] Python version: {sys.version}")
        logger.error(f"[OLLAMA-{request_id}] Platform: {sys.platform}")
        
        # Include relevant environment variables (without exposing secrets)
        env_info = {
            "OLLAMA_URL": os.getenv("OLLAMA_URL", "Not set"),
            "PATH_EXISTS": os.path.exists(os.getenv("OLLAMA_URL", "").replace("http://", "").split(":")[0] if os.getenv("OLLAMA_URL") else "")
        }
        logger.error(f"[OLLAMA-{request_id}] Environment info: {json.dumps(env_info)}")
        
        return f"ERROR-1000: Failed to generate response. Error: {str(e)}\n\nTraceback: {trace[:500]}..."

def format_extracted_sections(sections: List[Dict]) -> str:
    """Format extracted sections into a readable document"""
    import traceback
    import uuid
    
    task_id = str(uuid.uuid4())[:8]
    logger.info(f"[FORMAT-{task_id}] Starting section formatting for {len(sections)} sections")
    
    formatted_text = ""
    
    for i, section in enumerate(sections):
        try:
            section_id = f"{task_id}-S{i+1}"
            logger.info(f"[FORMAT-{section_id}] Processing section: {section['title']}")
            
            # Format heading based on level
            level = section.get('level', 1)
            heading_prefix = '#' * level
            
            formatted_text += f"{heading_prefix} {section['title']}\n\n"
            logger.info(f"[FORMAT-{section_id}] Added heading with level {level}")
            
            # Add just the section text (without the heading)
            if 'text' in section and section['text']:
                logger.info(f"[FORMAT-{section_id}] Adding section text of length {len(section['text'])} chars")
                formatted_text += f"# Section Text\n\n"
                optimized_text = get_optimized_text(section['text'])
                logger.info(f"[FORMAT-{section_id}] Optimized text length: {len(optimized_text)} chars")
                formatted_text += f"{optimized_text}\n\n"
            else:
                logger.warning(f"[FORMAT-{section_id}] Section has no 'text' field or it's empty")

            # Add translated text
            logger.info(f"[FORMAT-{section_id}] Translating text to Korean")
            try:
                korean_text = translate_text(section['text'], 'KO')
                if korean_text:
                    logger.info(f"[FORMAT-{section_id}] Translation successful, length: {len(korean_text)} chars")
                    formatted_text += f"{korean_text}\n\n"
                else:
                    logger.error(f"[FORMAT-{section_id}] ERROR-3001: Translation failed, returned None")
                    formatted_text += f"Translation failed. Please check logs.\n\n"
            except Exception as trans_e:
                trace = ''.join(traceback.format_exception(type(trans_e), trans_e, trans_e.__traceback__))
                logger.error(f"[FORMAT-{section_id}] ERROR-3002: Translation error: {str(trans_e)}")
                logger.error(f"[FORMAT-{section_id}] Translation traceback: {trace}")
                formatted_text += f"Translation error: {str(trans_e)}\n\n"

            # Generate summary
            logger.info(f"[FORMAT-{section_id}] Generating summary")
            try:
                # Try OCI first, fall back to Ollama
                try:
                    logger.info(f"[FORMAT-{section_id}] Attempting OCI summarization")
                    summary_text = _get_oci_llm_response('Please summarize this text:', section['text'])
                    logger.info(f"[FORMAT-{section_id}] OCI summarization successful")
                except Exception as oci_e:
                    logger.error(f"[FORMAT-{section_id}] ERROR-3003: OCI summarization failed: {str(oci_e)}")
                    logger.info(f"[FORMAT-{section_id}] Falling back to Ollama summarization")
                    summary_text = _get_ollama_llm_response('Please summarize this text:', section['text'])
                    logger.info(f"[FORMAT-{section_id}] Ollama summarization successful")
                
                if summary_text:
                    if summary_text.startswith("ERROR-"):
                        logger.error(f"[FORMAT-{section_id}] ERROR-3004: Summarization returned error: {summary_text[:100]}")
                        formatted_text += f"# Summary\n\nSummarization failed with error: {summary_text}\n\n"
                    else:
                        logger.info(f"[FORMAT-{section_id}] Summarization successful, length: {len(summary_text)} chars")
                        formatted_text += f"# Summary\n\n{summary_text}\n\n"
                        
                        # Translate summary
                        logger.info(f"[FORMAT-{section_id}] Translating summary to Korean")
                        korean_summary = translate_text(summary_text, 'KO')
                        if korean_summary:
                            formatted_text += f"# Summary (Korean)\n\n{korean_summary}\n\n"
                        else:
                            formatted_text += f"# Summary (Korean)\n\nSummary translation failed\n\n"
                else:
                    logger.error(f"[FORMAT-{section_id}] ERROR-3005: Summarization returned None")
                    formatted_text += f"# Summary\n\nSummarization failed. Please check logs.\n\n"
            except Exception as summ_e:
                trace = ''.join(traceback.format_exception(type(summ_e), summ_e, summ_e.__traceback__))
                logger.error(f"[FORMAT-{section_id}] ERROR-3006: Summarization error: {str(summ_e)}")
                logger.error(f"[FORMAT-{section_id}] Summarization traceback: {trace}")
                formatted_text += f"# Summary\n\nError generating summary: {str(summ_e)}\n\n"

            # Generate terms explanation
            logger.info(f"[FORMAT-{section_id}] Generating terms explanation")
            try:
                terms_explanation_text = _get_ollama_llm_response(
                    'Please explain the terms in this text in a way that someone unfamiliar with LLMs can easily understand:', 
                    section['text']
                )
                
                if terms_explanation_text:
                    if terms_explanation_text.startswith("ERROR-"):
                        logger.error(f"[FORMAT-{section_id}] ERROR-3007: Terms explanation returned error: {terms_explanation_text[:100]}")
                        formatted_text += f"# Terms Explanation\n\nTerms explanation failed with error: {terms_explanation_text}\n\n"
                    else:
                        logger.info(f"[FORMAT-{section_id}] Terms explanation successful, length: {len(terms_explanation_text)} chars")
                        formatted_text += f"# Terms Explanation\n\n{terms_explanation_text}\n\n"
                        
                        # Translate terms explanation
                        logger.info(f"[FORMAT-{section_id}] Translating terms explanation to Korean")
                        korean_terms = translate_text(terms_explanation_text, 'KO')
                        if korean_terms:
                            formatted_text += f"# Terms Explanation (Korean)\n\n{korean_terms}\n\n"
                        else:
                            formatted_text += f"# Terms Explanation (Korean)\n\nTerms explanation translation failed\n\n"
                else:
                    logger.error(f"[FORMAT-{section_id}] ERROR-3008: Terms explanation returned None")
                    formatted_text += f"# Terms Explanation\n\nTerms explanation failed. Please check logs.\n\n"
            except Exception as terms_e:
                trace = ''.join(traceback.format_exception(type(terms_e), terms_e, terms_e.__traceback__))
                logger.error(f"[FORMAT-{section_id}] ERROR-3009: Terms explanation error: {str(terms_e)}")
                logger.error(f"[FORMAT-{section_id}] Terms explanation traceback: {trace}")
                formatted_text += f"# Terms Explanation\n\nError generating terms explanation: {str(terms_e)}\n\n"
                
            logger.info(f"[FORMAT-{section_id}] Section processing complete")
            
        except Exception as section_e:
            section_trace = ''.join(traceback.format_exception(type(section_e), section_e, section_e.__traceback__))
            logger.error(f"[FORMAT-{task_id}] ERROR-3000: Section processing error: {str(section_e)}")
            logger.error(f"[FORMAT-{task_id}] Section traceback: {section_trace}")
            formatted_text += f"# Error Processing Section\n\nAn error occurred while processing this section: {str(section_e)}\n\n"
    
    logger.info(f"[FORMAT-{task_id}] Formatting complete, total length: {len(formatted_text)} chars")
    return formatted_text.strip()

def get_optimized_text(text: str) -> str:
    """Optimize the text for the LLM by normalizing whitespace"""
    import uuid
    
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[OPTIMIZE-{request_id}] Optimizing text of length {len(text)} chars")
    
    try:
        # Replace tabs and newlines with spaces
        result = text.replace('\t', ' ').replace('\n', ' ')
        
        # Recursively replace multiple spaces with a single space
        space_count = result.count('  ')
        while '  ' in result:
            result = result.replace('  ', ' ')
        
        logger.info(f"[OPTIMIZE-{request_id}] Optimization complete. Removed {space_count} double spaces. New length: {len(result)} chars")
        return result.strip()
    except Exception as e:
        import traceback
        trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"[OPTIMIZE-{request_id}] ERROR-5000: Error optimizing text: {str(e)}")
        logger.error(f"[OPTIMIZE-{request_id}] Traceback: {trace}")
        return text  # Return original text on error

def translate_text(text: str, input_lang: str) -> str:
    """Translate text using DeepL"""
    import traceback
    import uuid
    
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[TRANSLATE-{request_id}] Starting translation to {input_lang}")
    
    try:
        # Check for API key
        deepl_api_key = os.getenv('DEEPL_API_KEY')
        if not deepl_api_key:
            logger.error(f"[TRANSLATE-{request_id}] ERROR-4001: DEEPL_API_KEY not found in environment variables")
            return "ERROR-4001: DEEPL_API_KEY not set. Please set the DEEPL_API_KEY environment variable."
        
        logger.info(f"[TRANSLATE-{request_id}] Creating Translator with API key: {deepl_api_key[:4]}...")
        translator = deepl.Translator(deepl_api_key)
        
        # Normalize text
        logger.info(f"[TRANSLATE-{request_id}] Normalizing text of length {len(text)} chars")
        whole_text = text.replace('\t', ' ').replace('\n', ' ')
        while '  ' in whole_text:
            whole_text = whole_text.replace('  ', ' ')
        logger.info(f"[TRANSLATE-{request_id}] Normalized text length: {len(whole_text)} chars")
        
        # Handle text length limitations
        if len(whole_text) > 5000:
            logger.warning(f"[TRANSLATE-{request_id}] Text exceeds 5000 chars, truncating to 5000 chars")
            whole_text = whole_text[:5000] + "..."
        
        # Make the API call
        logger.info(f"[TRANSLATE-{request_id}] Sending translation request to DeepL")
        translated_text = translator.translate_text(
            whole_text,
            target_lang=input_lang,
        )
        logger.info(f"[TRANSLATE-{request_id}] Translation successful")
        
        return translated_text.text
    except Exception as e:
        trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        logger.error(f"[TRANSLATE-{request_id}] ERROR-4000: Error translating text: {str(e)}")
        logger.error(f"[TRANSLATE-{request_id}] Translation traceback: {trace}")
        return f"ERROR-4000: Translation failed: {str(e)}"

def main():
    """Main execution function"""
    logger.info(f"Main execution function")

    # pdf_path = input("Enter the path to your PDF file: ").strip()
    pdf_path = "C:/Users/jmko7/Downloads/NIPS-2017-attention-is-all-you-need-Paper.pdf"

    parser = PDFDocumentParser()

    try:
        logger.info(f"\nProcessing {pdf_path}...")
        logger.info("This may take a few moments depending on the PDF complexity...")
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = results_dir / f"document_analysis_{timestamp}.md"

        with open(report_path, "w", encoding="utf-8") as f:
            # Write header
            f.write(f"# Document parsing Report\n\n")
            f.write(f"## File Information\n")
            f.write(f"- **Source PDF**: `{pdf_path}`\n")
            f.write(f"- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            try:
                # Extract text content
                content = parser.extract_text_content(pdf_path)

                if content:
                    # Write document structure
                    if content['title']:
                        f.write(f"## Document Title\n{content['title']}\n\n")
                    
                    if content['abstract']:
                        f.write(f"## Abstract\n{content['abstract']}\n\n")
                        korean_abstract = translate_text(content['abstract'], 'KO')
                        f.write(f"## Abstract (Korean)\n{korean_abstract}\n\n")

                    # Write page-by-page content
                    f.write("## Page-by-Page Content\n\n")
                    for page in content['pages']:
                        f.write(f"### Page {page['page_number']}\n")
                        f.write("<details>\n<summary>Click to expand page content</summary>\n\n")
                        f.write("```text\n")
                        f.write(page['text'])
                        f.write("\n```\n</details>\n\n")

                        tables = parser.extract_tables(pdf_path, page['page_number'])

                        if tables:
                            f.write(f"## Tables Found\n")
                            f.write(f"Found {len(tables)} tables in the document.\n\n")
                            
                            # Save tables and write details for each table
                            parser.save_tables(tables)
                            
                            for i, table in enumerate(tables, 1):
                                f.write(f"### Table {i} (Page {table['metadata']['page']})\n")
                                f.write(f"- **Rows**: {table['metadata']['rows']}\n")
                                f.write(f"- **Columns**: {table['metadata']['columns']}\n")
                                
                                # Write table content
                                f.write("\n**Table Content**:\n")
                                f.write(table['markdown'])
                                f.write("\n\n")
                        else:
                            f.write("## Tables\n")
                            f.write("No tables were found in the document.\n\n")

                    if content['references']:
                        f.write("## References\n")
                        for i, ref in enumerate(content['references'], 1):
                            f.write(f"{i}. {ref}\n")
                        f.write("\n")

                    # Common academic paper structure for the "Attention is All You Need" paper
                    subtitles = [
                        "1  Introduction", 
                        "2  Background", 
                        "3  Model Architecture",
                        "3.1  Encoder and Decoder Stacks",
                        "3.2  Attention",
                        "3.2.1  Scaled Dot-Product Attention",
                        "3.2.2  Multi-Head Attention",
                        "3.2.3  Applications of Attention in our Model",
                        "3.3  Position-wise Feed-Forward Networks",
                        "3.4  Embeddings and Softmax",
                        "3.5  Positional Encoding",
                        "4  Why Self-Attention",
                        "5  Training",
                        "5.1  Training Data and Batching",
                        "5.2  Hardware and Schedule",
                        "5.3  Optimizer",
                        "5.4  Regularization",
                        "6  Results",
                        "6.1  Machine Translation",
                        "6.2  Model Variations",
                        "7  Conclusion",
                        "References"
                    ]

                    # Split by known subtitles
                    sections = split_by_known_subtitles(content['raw_text'], subtitles)

                    # Format the sections
                    formatted_text = format_extracted_sections(sections)

                    # Write the formatted text
                    f.write(f"## Document Structure by Section\n\n")
                    f.write(formatted_text)
                    f.write("\n\n")
                    
            except Exception as e:
                f.write("## Error Report\n")
                f.write(f"```\n{str(e)}\n```\n\n")
            
            # Write footer
            f.write("\n---\n")
            f.write("*Report generated by PDF Document Parser*\n")

        print(f"\nReport saved to: {report_path}")

    except Exception as e:
        logger.error(f"\nError processing PDF: {str(e)}")
        
        # Write error report
        error_report_path = results_dir / f"error_report_{timestamp}.md"
        with open(error_report_path, "w", encoding="utf-8") as f:
            f.write("# PDF Processing Error Report\n\n")
            f.write(f"## File Information\n")
            f.write(f"- **Source PDF**: `{pdf_path}`\n")
            f.write(f"- **Error Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Error Details\n")
            f.write(f"```\n{str(e)}\n```\n")

if __name__ == "__main__":
    main() 