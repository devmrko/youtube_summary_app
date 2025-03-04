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

class EnhancedPDFTableParser:
    def __init__(self, output_dir: str = "parsed_tables"):
        """Initialize the Enhanced PDF Table Parser"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Special character mappings
        self.special_chars = {
            '제곱': '²',
            'squared': '²',
            'cubic': '³',
            'sum': '∑',
            'delta': 'Δ',
            'micro': 'μ',
            'degree': '°',
            'plusminus': '±',
            'times': '×',
            'divide': '÷',
            'alpha': 'α',
            'beta': 'β',
            'gamma': 'γ',
            'omega': 'Ω'
        }

        # Superscript number mappings
        self.superscript_map = {
            '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
            '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
            '-': '⁻', '+': '⁺', '=': '⁼', '(': '⁽', ')': '⁾',
            'n': 'ⁿ', 'i': 'ⁱ'
        }

        # Korean number mappings
        self.korean_numbers = {
            '영': '0', '일': '1', '이': '2', '삼': '3', '사': '4',
            '오': '5', '육': '6', '칠': '7', '팔': '8', '구': '9',
            '십': '10'
        }
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Install required dependencies
        self._install_dependencies()

    def _install_dependencies(self):
        """Install required dependencies"""
        try:
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
            # Configure Tesseract for better number recognition
            custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789.×^*·E() "' 
            return pytesseract.image_to_string(image, config=custom_config)
        except Exception as e:
            self.logger.error(f"OCR error: {str(e)}")
            return ""

    def _process_cell_with_ocr(self, cell_image: Image.Image) -> str:
        """Process a single cell with OCR and apply scientific notation formatting"""
        try:
            # Perform OCR on the cell
            text = self._perform_ocr(cell_image)
            
            # Clean and normalize the text
            text = text.strip()
            text = self._normalize_scientific_notation(text)
            text = self._normalize_special_chars(text)
            
            return text
        except Exception as e:
            self.logger.error(f"Cell OCR processing error: {str(e)}")
            return ""

    def extract_tables(self, pdf_path: str) -> List[Dict]:
        """Extract tables using Tabula's detection with OCR enhancement"""
        tables = []
        
        try:
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
                            self.logger.info(f"Extracted table from page {page_num}")
            
            return tables
            
        except Exception as e:
            self.logger.error(f"Error extracting tables: {str(e)}")
            raise

    def _enhance_table_with_ocr(self, df: pd.DataFrame, page_image: Image.Image, table_area: Dict) -> pd.DataFrame:
        """Enhance table data with OCR processing"""
        try:
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

    def _normalize_special_chars(self, text: str) -> str:
        """Normalize special characters and mathematical symbols with improved scientific notation handling"""
        if not isinstance(text, str):
            return text
            
        # Convert full-width characters to half-width
        text = unicodedata.normalize('NFKC', text)
        
        # Handle scientific notation patterns
        def handle_scientific_notation(match):
            base = match.group(1)
            power = match.group(2)
            # Convert to proper scientific notation format
            return f"{base}×10{self._to_superscript(power)}"
        
        # Scientific notation patterns
        scientific_patterns = [
            (r'(\d+\.?\d*)\s*[·*]\s*10\s*(?:squared|제곱)?\s*(\d+)', handle_scientific_notation),  # 2.3 * 10 squared 19
            (r'(\d+\.?\d*)\s*[·*]\s*10\^(\d+)', handle_scientific_notation),  # 2.3 * 10^19
            (r'(\d+\.?\d*)[Ee](\d+)', handle_scientific_notation),  # 2.3E19
        ]
        
        for pattern, handler in scientific_patterns:
            text = re.sub(pattern, handler, text)
        
        # Handle multiplication symbols
        text = re.sub(r'\*', '×', text)  # Replace * with ×
        text = re.sub(r'·', '×', text)   # Replace · with ×
        
        # Handle Korean number + 제곱 pattern
        for kor_num, digit in self.korean_numbers.items():
            text = text.replace(f"{kor_num}제곱", self._to_superscript(digit))
        
        # Handle "제곱" with preceding number
        text = re.sub(r'(\d+)제곱', lambda m: self._to_superscript(m.group(1)), text)
        
        # Handle number followed by "승"
        text = re.sub(r'(\d+)승', lambda m: self._to_superscript(m.group(1)), text)
        
        # Handle other special character words
        for word, symbol in self.special_chars.items():
            text = text.replace(word, symbol)
        
        return text

    def _to_superscript(self, number: str) -> str:
        """Convert a number string to superscript"""
        return ''.join(self.superscript_map[d] for d in str(number))

    def _process_table(self, df: pd.DataFrame, page_num: int) -> Dict:
        """Process extracted table and improve its structure"""
        try:
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
        if row.empty:
            return False
        
        # Check header characteristics
        non_empty_cells = row.astype(str).str.strip().str.len() > 0
        short_cells = row.astype(str).str.len() < 50
        non_numeric = ~row.astype(str).str.replace('.', '').str.isdigit()
        
        return (non_empty_cells & short_cells & non_numeric).mean() > 0.8

    def _check_special_chars(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Check for special characters in the DataFrame"""
        special_chars_found = {'mathematical': [], 'scientific': [], 'other': []}
        
        # Regular expressions for different types of special characters
        math_pattern = r'[²³∑±×÷⁰¹²³⁴⁵⁶⁷⁸⁹]'
        scientific_pattern = r'[αβγμΩΔ°]'
        
        for col in df.columns:
            for value in df[col].astype(str):
                if re.search(math_pattern, value):
                    special_chars_found['mathematical'].extend(re.findall(math_pattern, value))
                if re.search(scientific_pattern, value):
                    special_chars_found['scientific'].extend(re.findall(scientific_pattern, value))
                
                # Find other unicode special characters
                other_chars = [c for c in value if unicodedata.category(c).startswith(('So', 'Sm'))]
                if other_chars:
                    special_chars_found['other'].extend(other_chars)
        
        # Remove duplicates
        special_chars_found = {k: list(set(v)) for k, v in special_chars_found.items()}
        return special_chars_found

    def _normalize_scientific_notation(self, text: str) -> str:
        """Pre-process scientific notation before general special character handling"""
        if not isinstance(text, str):
            return text
        
        # Pattern for scientific notation with various formats
        patterns = [
            # 2.3 * 10 squared 19
            r'(\d+\.?\d*)\s*[·*]\s*10\s*(?:squared|제곱)?\s*(\d+)',
            # 2.3 * 10^19
            r'(\d+\.?\d*)\s*[·*]\s*10\^(\d+)',
            # 2.3E19
            r'(\d+\.?\d*)[Ee](\d+)',
            # Handle concatenated format like 1019 -> 10¹⁹
            r'10(\d{2,})',  # Match '10' followed by 2 or more digits
            # Handle format like 2.3·1019 -> 2.3×10¹⁹
            r'(\d+\.?\d*)[·*]10(\d{2,})'
        ]
        
        for pattern in patterns:
            if pattern.startswith(r'10(\d{2,})'):
                # Special handling for concatenated format
                text = re.sub(pattern, lambda m: f"10{self._to_superscript(m.group(1))}", text)
            elif pattern.startswith(r'(\d+\.?\d*)[·*]10(\d{2,})'):
                # Special handling for number·10XX format
                text = re.sub(pattern, lambda m: f"{m.group(1)}×10{self._to_superscript(m.group(2))}", text)
            else:
                # Standard scientific notation handling
                text = re.sub(pattern, 
                            lambda m: f"{m.group(1)}×10{self._to_superscript(m.group(2))}",
                            text)
        
        return text

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
                
                self.logger.info(f"Saved table to {table_dir}")
                
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
            
            # Special character mapping for text extraction
            special_char_map = {
                '¹': ' power 1 ',
                '²': ' power 2 ',
                '³': ' power 3 ',
                '⁴': ' power 4 ',
                '⁵': ' power 5 ',
                '⁶': ' power 6 ',
                '⁷': ' power 7 ',
                '⁸': ' power 8 ',
                '⁹': ' power 9 ',
                '⁰': ' power 0 ',
                '×': ' times ',
                '·': ' times ',
                '*': ' times ',
                '^': ' power ',
                '±': ' plus-minus ',
                'Δ': ' delta ',
                'μ': ' micro ',
                'α': ' alpha ',
                'β': ' beta ',
                'γ': ' gamma ',
                'Ω': ' omega ',
                '∑': ' sum ',
                '°': ' degrees ',
            }
            
            def process_special_chars(text):
                """Process special characters in text"""
                # First handle scientific notation patterns
                text = re.sub(r'(\d+)(?:E|e|×10|x10|·10)(\d+)', 
                            lambda m: f"{m.group(1)} times 10 power {m.group(2)}", 
                            text)
                
                # Handle superscript numbers after 10
                text = re.sub(r'10([⁰¹²³⁴⁵⁶⁷⁸⁹]+)', 
                            lambda m: f"10 power {''.join(special_char_map.get(c, c).strip() for c in m.group(1))}", 
                            text)
                
                # Replace other special characters
                for char, replacement in special_char_map.items():
                    text = text.replace(char, replacement)
                
                # Clean up multiple spaces
                text = re.sub(r'\s+', ' ', text)
                return text
            
            # First try with pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
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
                        word_text = process_special_chars(word['text'])
                        
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

def main():
    """Main execution function"""
    parser = EnhancedPDFTableParser()
    
    pdf_path = input("Enter the path to your PDF file: ").strip()
    
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