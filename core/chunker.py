from .config import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP

def chunk_text(pages_data, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_OVERLAP):
    """
    Splits PDF pages into overlapping text chunks.

    Args:
        pages_data (list): List of page dictionaries.
        chunk_size (int): Maximum characters per chunk.
        overlap (int): Overlap between chunks.
    Returns:
        tuple: (chunks, metadatas)
    """
    chunks = []
    metadatas = []
    separators = ["\n\n", "\n", ". ", " ", ""]

    for item in pages_data:
        text = item["text"]
        page_num = item["page"]
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            
            if end < len(text):
                for sep in separators:
                    last_sep_idx = text.rfind(sep, start, end)
                    if last_sep_idx != -1 and last_sep_idx > start:
                        end = last_sep_idx + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
                metadatas.append({"page": page_num})

            # Prevent infinite loop if overlap >= chunk length
            next_start = end - overlap

            if next_start <= start:
                start = end
            else:
                start = next_start
    
    return chunks, metadatas