"""
PDF Loading Module

This module handles extraction and basic cleaning of text from PDF files
using the `unstructured` library. It removes headers, footers, and page numbers,
and returns page-wise structured text suitable for downstream processing
(e.g., chunking and embedding).
"""

from typing import List, Dict
from unstructured.partition.pdf import partition_pdf
import logging

# Silence verbose logs
logging.getLogger("unstructured").setLevel(logging.ERROR)
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def load_pdf_text(pdf_path: str) -> List[Dict[str, str]]:
    """
    Extracts cleaned, page-wise text content from a PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        List[Dict[str, str]]: A list of dictionaries:
            [
                {"text": "...", "page": 1},
                {"text": "...", "page": 2}
            ]
    """
    try:
        elements = partition_pdf(
            filename=pdf_path,
            include_page_breaks=True,
            strategy="fast",
            languages=["eng"]
        )

        pages_data: List[Dict[str, str]] = []
        current_page_text: List[str] = []
        current_page_number = 1

        excluded_types = ["Header", "Footer", "PageNumber"]

        for el in elements:
            if el.category == "PageBreak":
                if current_page_text:
                    pages_data.append({
                        "text": "\n".join(current_page_text),
                        "page": current_page_number
                    })
                    current_page_text = []
                current_page_number += 1
                continue

            if el.category not in excluded_types and el.text:
                current_page_text.append(el.text)

        if current_page_text:
            pages_data.append({
                "text": "\n".join(current_page_text),
                "page": current_page_number
            })

        return pages_data

    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []