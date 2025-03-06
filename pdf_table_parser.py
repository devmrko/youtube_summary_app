"""
Enhanced PDF Table Parser using Tabula with OCR and special character support
This script provides better quality table extraction from PDFs
"""

import os
import pandas as pd
import tabula
import re
from pathlib import Path
from typing import List, Dict
import logging
from datetime import datetime
from tabulate import tabulate
import numpy as np
import json
import openpyxl
import sys
import unicodedata
import pdf2image
import pytesseract
from PIL import Image
import io
import oci

MODEL_ID = "ocid1.generativeaimodel.oc1.us-chicago-1.amaaaaaask7dceyarleil5jr7k2rykljkhapnvhrqvzx4cwuvtfedlfxet4q"
logger = logging.getLogger(__name__)

class EnhancedPDFTableParser:

    def __init__(self, output_dir: str = "parsed_tables"):
        """Initialize the Enhanced PDF Table Parser"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize OCI configuration
        try:
            print(f"# Initialize OCI configuration")

            self.config = oci.config.from_file()

            self.llm_client = oci.generative_ai_inference.GenerativeAiInferenceClient(
            config=self.config,
            service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
            retry_strategy=oci.retry.NoneRetryStrategy(),
            timeout=3600  # Single timeout value in seconds (1 hour)
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to load OCI config from file: {str(e)}")
            self.config = {
                "compartment_id": os.getenv("OCI_COMPARTMENT_ID"),
                "tenancy": os.getenv("OCI_TENANCY"),
                "user": os.getenv("OCI_USER"),
                "key_file": os.getenv("OCI_KEY_FILE"),
                "fingerprint": os.getenv("OCI_FINGERPRINT"),
                "region": os.getenv("OCI_REGION", "us-ashburn-1")
            }
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Install required dependencies
        self._install_dependencies()

    # Install required dependencies about OCR
    def _install_dependencies(self):
        """Install required dependencies"""
        try:
            print(f"# Initialize dependencies for OCR")
            import jpype
            import pytesseract
            import pdf2image
        except ImportError:
            print("Installing required dependencies...")
            os.system(f"{sys.executable} -m pip install JPype1 pytesseract pdf2image Pillow")
            print("Please also install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
            print("And add it to your system PATH")
            print("Please restart the script after installation.")
            sys.exit(1)

    def _perform_ocr(self, image: Image.Image) -> str:
        """Perform OCR on an image"""
        try:
            print(f"# Perform OCR on an image")
            # Configure Tesseract for better number recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789.×^*·E() "' 
            return pytesseract.image_to_string(image, config=custom_config)
        except Exception as e:
            self.logger.error(f"OCR error: {str(e)}")
            return ""

    def extract_tables(self, pdf_path: str) -> List[Dict]:
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
            
            # Convert PDF pages to images with explicit poppler path
            images = pdf2image.convert_from_path(pdf_path, poppler_path=poppler_path)
            
            # Extract tables using Tabula for structure
            dfs = tabula.read_pdf(
                pdf_path,
                pages='all',
                multiple_tables=True,
                guess=True,
                encoding='utf-8',
                java_options=['-Dfile.encoding=UTF8'],
                pandas_options={
                    'header': None,
                    'encoding': 'utf-8'
                }
            )
            
            # Process each table with OCR enhancement
            for page_num, (df, page_image) in enumerate(zip(dfs, images), 1):
                if not df.empty:
                    # Get table bounds from Tabula
                    table_areas = tabula.read_pdf(
                        pdf_path,
                        pages=page_num,
                        multiple_tables=True,
                        guess=True,
                        encoding='utf-8',
                        output_format='json'
                    )
                    
                    # Process each cell with OCR
                    for table_area in table_areas:
                        df_enhanced = self._enhance_table_with_ocr(df, page_image, table_area)
                        processed_table = self._process_table(df_enhanced, page_num)
                        if processed_table:
                            tables.append(processed_table)
                            print(f"Extracted table from page {page_num}")
            
            return tables
            
        except Exception as e:
            self.logger.error(f"Error extracting tables: {str(e)}")
            raise

    def _enhance_table_with_ocr(self, df: pd.DataFrame, page_image: Image.Image, table_area: Dict) -> pd.DataFrame:
        """Enhance table data with OCR processing"""
        try:
            print(f"# Enhance table data with OCR processing")
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
                    cell_value = str(df_enhanced.iloc[i, j])
                    
                    # Check if cell might contain scientific notation
                    if re.search(r'\d+\s*[·*]?\s*10?\d+', cell_value):
                        # Calculate cell position and crop
                        cell_height = (bottom - top) / len(df_enhanced)
                        cell_width = (right - left) / len(df_enhanced.columns)
                        cell_top = top + (i * cell_height)
                        cell_left = left + (j * cell_width)
                        cell_image = page_image.crop((
                            cell_left, cell_top,
                            cell_left + cell_width,
                            cell_top + cell_height
                        ))
                        
                        # Process cell with OCR
                        ocr_text = self._process_cell_with_ocr(cell_image)
                        if ocr_text:
                            df_enhanced.iloc[i, j] = ocr_text
            
            return df_enhanced
            
        except Exception as e:
            self.logger.error(f"Error enhancing table with OCR: {str(e)}")
            return df

    def _process_table(self, df: pd.DataFrame, page_num: int) -> Dict:
        """Process extracted table and improve its structure"""
        try:
            print(f"# Process extracted table and improve its structure")
            # Pre-process scientific notation before general special characters
            df = df.applymap(lambda x: self._normalize_scientific_notation(str(x)) if pd.notnull(x) else x)
            
            # Handle encoding and special characters for DataFrame
            df = df.applymap(lambda x: self._normalize_special_chars(str(x)) if pd.notnull(x) else x)
            
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
                    'columns': len(df.columns),
                    'has_special_chars': self._check_special_chars(df)
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

    def save_tables(self, tables: List[Dict]) -> None:
        """Save tables in multiple formats with improved organization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, table in enumerate(tables, 1):
            try:
                print(f"# Save tables in multiple formats with improved organization")
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
                
                # Save as HTML with MathJax support
                html_path = table_dir / "table.html"
                html_content = f"""
                <html>
                <head>
                    <meta charset="UTF-8">
                    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
                    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
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
                
                # Save as LaTeX if special characters are found
                if any(table['metadata'].get('has_special_chars', {}).values()):
                    latex_path = table_dir / "table.tex"
                    with open(latex_path, 'w', encoding='utf-8') as f:
                        f.write(table['latex'])
                
                # Save metadata with special character information
                meta_path = table_dir / "metadata.json"
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(table['metadata'], f, indent=4, ensure_ascii=False)
                
                print(f"Saved table to {table_dir}")
                
            except Exception as e:
                self.logger.error(f"Error saving table {i}: {str(e)}")

    def _format_excel(self, writer: pd.ExcelWriter, df: pd.DataFrame) -> None:
        """Apply formatting to Excel output"""
        worksheet = writer.sheets['Table']
        
        # Format header
        for col in range(len(df.columns)):
            cell = worksheet.cell(row=1, column=col+1)
            cell.font = openpyxl.styles.Font(bold=True)
            cell.fill = openpyxl.styles.PatternFill(start_color='F2F2F2', end_color='F2F2F2', fill_type='solid')

    def extract_text_content(self, pdf_path: str) -> Dict:
        """Extract text content and structure from PDF while preserving original spacing and special characters"""
        try:
            import pdfplumber
            
            content = {
                'title': '',
                'abstract': '',
                'sections': [],
                'references': [],
                'raw_text': '',
                'pages': []
            }
            
            # First try with pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                print(f"# Extract text content and structure from PDF while preserving original spacing and special characters")
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
                
                # Find sections with preserved spacing
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

    def _process_markdown_with_summaries(self, markdown_path: str, deepl_api_key: str = None) -> Dict:
        """Process markdown report to add summaries and translations"""
        try:
            print(f"# Process markdown report to add summaries and translations")
            import deepl
            import oci
            from oci.generative_ai_inference import GenerativeAiInferenceClient
            from oci.generative_ai_inference.models import (
                TextContent, Message, GenericChatRequest,
                ChatDetails, OnDemandServingMode
            )

            summaries = {
                'sections': [],
                'tables': [],
                'overall': ''
            }

            # Initialize DeepL translator if API key is provided
            translator = None
            if deepl_api_key:
                translator = deepl.Translator(deepl_api_key)

            # Initialize OCI Generative AI client
            ai_client = GenerativeAiInferenceClient(self.config)

            # Read markdown content
            with open(markdown_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split content into sections
            sections = re.split(r'\n##\s+', content)
            
            # Process each section
            for section in sections:
                if not section.strip():
                    continue
                
                # Get section title and content
                lines = section.split('\n')
                title = lines[0].strip('# ')
                content = '\n'.join(lines[1:]).strip()

                # Generate summary using OCI Generative AI
                summary = self._get_oci_summary(content, ai_client)
                
                # Translate if DeepL is available
                korean_summary = None
                korean_content = None
                if translator:
                    try:
                        korean_summary = translator.translate_text(
                            summary,
                            target_lang="KO",
                        ).text
                        korean_content = translator.translate_text(
                            content,
                            target_lang="KO",
                        ).text
                        
                    except Exception as e:
                        self.logger.error(f"Translation error: {str(e)}")

                summaries['sections'].append({
                    'title': title,
                    'content': content,
                    'summary': summary,
                    'korean_summary': korean_summary,
                    'korean_content': korean_content
                })

            # Generate overall summary
            overall_text = '\n'.join(section['content'] for section in summaries['sections'])
            summaries['overall'] = self._get_oci_summary(overall_text[:5000], ai_client)  # Limit to first 5000 chars
            
            if translator:
                try:
                    summaries['overall_korean'] = translator.translate_text(
                        summaries['overall'],
                        target_lang="KO",
                    ).text
                except Exception as e:
                    self.logger.error(f"Translation error: {str(e)}")

            return summaries

        except Exception as e:
            self.logger.error(f"Error processing markdown with summaries: {str(e)}")
            return None

    def _get_oci_summary(self, text: str, ai_client: 'GenerativeAiInferenceClient') -> str:
        """Generate summary using OCI Generative AI"""
        try:
            print(f"# Generate summary using OCI Generative AI")
            # Create text content
            content = oci.generative_ai_inference.models.TextContent()
            content.text = f"Please summarize this text:\n\n{text}"
            content.type = "TEXT"
            
            # Create message
            message = oci.generative_ai_inference.models.Message()
            message.role = "USER"
            message.content = [content]
            
            # Create chat request
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
            
            for key, value in model_params.items():
                setattr(chat_request, key, value)

            # Create chat details
            chat_details = oci.generative_ai_inference.models.ChatDetails()
            chat_details.serving_mode = oci.generative_ai_inference.models.OnDemandServingMode(
                model_id=MODEL_ID
            )
            chat_details.chat_request = chat_request
            chat_details.compartment_id = oci.config.from_file()["tenancy"]

            # Get response
            # response = ai_client.chat(chat_details)
            # summary = response.data.chat_response.choices[0].message.content[0].text

            response = self.llm_client.chat(chat_details)
            return response.data.chat_response.choices[0].message.content[0].text

            # return summary[:500]  # Limit summary to 500 chars
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return text[:500]  # Return truncated text if summarization fails

def main():
    """Main execution function"""
    print(f"# Main execution function")
    parser = EnhancedPDFTableParser()
    
    pdf_path = input("Enter the path to your PDF file: ").strip()
    deepl_api_key = os.getenv('DEEPL_API_KEY')
    
    if not os.path.exists(pdf_path):
        print("Error: PDF file not found!")
        return
    
    try:
        print(f"\nProcessing {pdf_path}...")
        print("This may take a few moments depending on the PDF complexity...")
        
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create markdown report
        report_path = results_dir / f"document_analysis_{timestamp}.md"
        
        with open(report_path, "w", encoding="utf-8") as f:
            # Write header
            f.write(f"# Document Analysis Report\n\n")
            f.write(f"## File Information\n")
            f.write(f"- **Source PDF**: `{pdf_path}`\n")
            f.write(f"- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            try:
                # Extract text content
                content = parser.extract_text_content(pdf_path)
                
                if content:
                    # Write document structure
                    if content['title']:
                        f.write(f"## Document Title\n{content['title']}\n\n")
                    
                    if content['abstract']:
                        f.write(f"## Abstract\n{content['abstract']}\n\n")
                    
                    if content['sections']:
                        f.write("## Document Structure\n\n")
                        for section in content['sections']:
                            f.write(f"### {section['title']}\n")
                            f.write(f"{section['content']}\n\n")
                    
                    # Write complete raw text
                    f.write("## Complete Text Content\n\n")
                    f.write("<details>\n<summary>Click to expand full text</summary>\n\n")
                    f.write("```text\n")
                    f.write(content['raw_text'])
                    f.write("\n```\n</details>\n\n")
                    
                    # Write page-by-page content
                    f.write("## Page-by-Page Content\n\n")
                    for page in content['pages']:
                        f.write(f"### Page {page['page_number']}\n")
                        f.write("<details>\n<summary>Click to expand page content</summary>\n\n")
                        f.write("```text\n")
                        f.write(page['text'])
                        f.write("\n```\n</details>\n\n")
                    
                    if content['references']:
                        f.write("## References\n")
                        for i, ref in enumerate(content['references'], 1):
                            f.write(f"{i}. {ref}\n")
                        f.write("\n")
                
                # Extract and process tables
                tables = parser.extract_tables(pdf_path)
                
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
                    
                # Add summaries and translations section
                print("\nGenerating summaries and translations...")
                summaries = parser._process_markdown_with_summaries(report_path, deepl_api_key)
                
                if summaries:
                    f.write("\n## Document Summary and Translation\n\n")
                    
                    # Write overall summary
                    f.write("### Overall Summary\n\n")
                    f.write("#### English Summary\n")
                    f.write(f"{summaries['overall']}\n\n")
                    
                    if 'overall_korean' in summaries:
                        f.write("#### Korean Summary (한국어 요약)\n")
                        f.write(f"{summaries['overall_korean']}\n\n")
                    
                    # Write section summaries
                    f.write("### Section Summaries\n\n")
                    for section in summaries['sections']:
                        f.write(f"#### {section['title']}\n\n")
                        
                        f.write("##### English Summary\n")
                        f.write(f"{section['summary']}\n\n")
                        
                        if section['korean_summary']:
                            f.write("##### Korean Summary (한국어 요약)\n")
                            f.write(f"{section['korean_summary']}\n\n")
                        
                        f.write("<details>\n<summary>Full Korean Translation (전체 한국어 번역)</summary>\n\n")
                        if section['korean_content']:
                            f.write(f"{section['korean_content']}\n")
                        f.write("</details>\n\n")
                
            except Exception as e:
                f.write("## Error Report\n")
                f.write(f"An error occurred during document analysis:\n")
                f.write(f"```\n{str(e)}\n```\n\n")
            
            # Write footer
            f.write("\n---\n")
            f.write("*Report generated by Enhanced PDF Document Parser*\n")
        
        print(f"\nReport saved to: {report_path}")
        
    except Exception as e:
        print(f"\nError processing PDF: {str(e)}")
        
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