import logging
import os
from langchain.tools import tool

logger = logging.getLogger(__name__)

@tool
def extract_cv_text(file_path: str) -> str:
    """Extract raw text content from a CV file (PDF or TXT).

    Args:
        file_path: Path to the CV file.

    Returns:
        The raw text content extracted from the CV. Returns an empty string on error.

    Raises:
        FileNotFoundError: If the CV file does not exist.

    Example:
        >>> extract_cv_text("sample_resume.pdf")
        'John Doe\\nSoftware Engineer...'
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CV file not found: '{file_path}'")

    file_ext = os.path.splitext(file_path)[1].lower()

    if file_ext == '.pdf':
        try:
            import pdfplumber
            text_parts = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            
            final_text = "\n".join(text_parts).strip()
            logger.info(f"Extracted {len(final_text)} characters from PDF: {file_path}")
            return final_text
        except ImportError:
            logger.error("pdfplumber not installed.")
            return ""
        except Exception as e:
            logger.error(f"Error extracting PDF {file_path}: {e}")
            return ""

    elif file_ext == '.txt':
        encodings = ['utf-8', 'latin-1', 'cp1252']
        for enc in encodings:
            try:
                with open(file_path, 'r', encoding=enc) as f:
                    final_text = f.read().strip()
                logger.info(f"Extracted {len(final_text)} characters from TXT: {file_path} using encoding {enc}")
                return final_text
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading TXT {file_path} with encoding {enc}: {e}")
                return ""
        
        logger.error(f"Failed to extract TXT text from {file_path} with all attempted encodings.")
        return ""
    
    else:
        logger.error(f"Unsupported file type: {file_ext}")
        return ""
