from typing import Tuple, BinaryIO
import PyPDF2
from io import BytesIO


class DocumentProcessor:
    """Service for processing and extracting text from documents."""
    
    @staticmethod
    def extract_text(file: BinaryIO, filename: str) -> Tuple[str, str]:
        """
        Extract text from uploaded file.
        
        Args:
            file: File object (from FastAPI UploadFile)
            filename: Original filename
            
        Returns:
            Tuple of (extracted_text, file_type)
            
        Raises:
            ValueError: If file type is not supported
        """
        file_type = DocumentProcessor._get_file_type(filename)
        
        if file_type == "pdf":
            text = DocumentProcessor._extract_from_pdf(file)
        elif file_type == "txt":
            text = DocumentProcessor._extract_from_txt(file)
        else:
            raise ValueError(
                f"Unsupported file type: {file_type}. "
                "Supported types: .pdf, .txt"
            )
        
        return text, file_type
    
    @staticmethod
    def _get_file_type(filename: str) -> str:
        """
        Get file type from filename.
        
        Args:
            filename: File name with extension
            
        Returns:
            File type (pdf, txt)
        """
        extension = filename.lower().split('.')[-1]
        
        if extension == "pdf":
            return "pdf"
        elif extension == "txt":
            return "txt"
        else:
            raise ValueError(f"Unsupported file extension: .{extension}")
    
    @staticmethod
    def _extract_from_pdf(file: BinaryIO) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file: PDF file object
            
        Returns:
            Extracted text
        """
        try:
            # Read file content
            content = file.read()
            pdf_file = BytesIO(content)
            
            # Create PDF reader
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from all pages
            text_parts = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():
                    text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            
            if not full_text.strip():
                raise ValueError("No text could be extracted from PDF")
            
            return full_text
            
        except Exception as e:
            raise ValueError(f"Error extracting text from PDF: {str(e)}")
    
    @staticmethod
    def _extract_from_txt(file: BinaryIO) -> str:
        """
        Extract text from TXT file.
        
        Args:
            file: Text file object
            
        Returns:
            Extracted text
        """
        try:
            # Read file content with UTF-8 encoding
            content = file.read()
            
            # Try UTF-8 first, fallback to latin-1
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                text = content.decode('latin-1')
            
            if not text.strip():
                raise ValueError("Text file is empty")
            
            return text
            
        except Exception as e:
            raise ValueError(f"Error extracting text from TXT: {str(e)}")
    
    @staticmethod
    def validate_file(filename: str, file_size: int, max_size_mb: int = 10) -> None:
        """
        Validate uploaded file.
        
        Args:
            filename: File name
            file_size: File size in bytes
            max_size_mb: Maximum allowed file size in MB
            
        Raises:
            ValueError: If validation fails
        """
        # Check file type
        DocumentProcessor._get_file_type(filename)
        
        # Check file size
        max_size_bytes = max_size_mb * 1024 * 1024
        if file_size > max_size_bytes:
            raise ValueError(
                f"File size ({file_size / 1024 / 1024:.2f} MB) exceeds "
                f"maximum allowed size ({max_size_mb} MB)"
            )
        
        if file_size == 0:
            raise ValueError("File is empty")