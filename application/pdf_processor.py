"""
PDF processing module for extracting text from uploaded PDF files.
Note: This is a simplified version. For production use, install PyPDF2.
"""
from typing import List, Dict
import io


class PDFProcessor:
    """Handle PDF file processing and text extraction."""
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from a PDF file.
        
        Args:
            pdf_file: File-like object containing PDF data
            
        Returns:
            Extracted text as a string
        """
        try:
            # Try to use PyPDF2 if available
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text_parts = []
            
            for page in pdf_reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            
            return "\n".join(text_parts)
        except ImportError:
            # Fallback: treat as text file for testing
            # In production, PyPDF2 should be installed
            try:
                pdf_file.seek(0)
                content = pdf_file.read()
                if isinstance(content, bytes):
                    return content.decode('utf-8', errors='ignore')
                return str(content)
            except Exception as e:
                raise ValueError(f"PyPDF2 not installed and fallback failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")
    
    def process_multiple_pdfs(self, pdf_files: List) -> Dict[str, str]:
        """Process multiple PDF files and extract text from each.
        
        Args:
            pdf_files: List of file-like objects containing PDF data
            
        Returns:
            Dictionary mapping PDF filename to extracted text
        """
        results = {}
        
        for pdf_file in pdf_files:
            filename = getattr(pdf_file, 'filename', f'pdf_{len(results)}')
            
            # Reset file pointer to beginning
            pdf_file.seek(0)
            
            text = self.extract_text_from_pdf(pdf_file)
            results[filename] = text
        
        return results
