import re
import json
import logging
from bs4 import BeautifulSoup
from typing import List
from langchain.schema import Document

ZERO_WIDTH_CHARS = [
    '\u200B',  # zero width space
    '\u200C',  # zero width non-joiner
    '\u200D',  # zero width joiner
    '\uFEFF',  # zero width no-break space (BOM)
]

def detect_zero_width(text: str):
    positions = []
    for zchar in ZERO_WIDTH_CHARS:
        start = 0
        while True:
            idx = text.find(zchar, start)
            if idx == -1:
                break
            positions.append({'char': zchar, 'index': idx})
            start = idx + 1
    return positions

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    try:
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    except:
        return None

def color_distance(c1, c2):
    return sum((a-b)**2 for a, b in zip(c1, c2))**0.5

def parse_color(color_str):
    color_str = color_str.strip().lower()
    if color_str.startswith('#'):
        return hex_to_rgb(color_str)
    elif color_str.startswith('rgb'):
        try:
            nums = re.findall(r'\d+', color_str)
            if len(nums) == 3:
                return tuple(int(n) for n in nums)
        except:
            return None
    return None

def detect_invisible_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    alerts = []
    COLOR_THRESHOLD = 30

    for tag in soup.find_all(style=True):
        style = tag['style']
        color_match = re.search(r'color\s*:\s*([^;]+)', style, re.I)
        bgcolor_match = re.search(r'background-color\s*:\s*([^;]+)', style, re.I)

        text_color = parse_color(color_match.group(1)) if color_match else None
        bg_color = parse_color(bgcolor_match.group(1)) if bgcolor_match else (255, 255, 255)

        if text_color and bg_color:
            dist = color_distance(text_color, bg_color)
            if dist < COLOR_THRESHOLD:
                alerts.append({
                    'tag': str(tag)[:50],
                    'text_color': color_match.group(1),
                    'bg_color': bgcolor_match.group(1) if bgcolor_match else 'default (white)',
                    'distance': dist,
                })

    return alerts

def detect_small_font(html):
    soup = BeautifulSoup(html, 'html.parser')
    alerts = []
    FONT_SIZE_THRESHOLD = 6

    for tag in soup.find_all(style=True):
        style = tag['style']
        font_size_match = re.search(r'font-size\s*:\s*([^;]+)', style, re.I)
        if font_size_match:
            size_str = font_size_match.group(1).strip()
            size_match = re.match(r'(\d+)(px|pt|em)?', size_str)
            if size_match:
                size_val = int(size_match.group(1))
                if size_val <= FONT_SIZE_THRESHOLD:
                    alerts.append({
                        'tag': str(tag)[:50],
                        'font_size': size_str,
                    })

    return alerts

def extract_inline_html_from_md(md_text):
    pattern = re.compile(r'(<[a-zA-Z][^>]*>.*?<\/[a-zA-Z]+>)', re.DOTALL)
    return pattern.findall(md_text)

def detect_prompt_injection(document: Document):
    result = {
        'zero_width': [],
        'invisible_text': [],
        'small_font': [],
    }

    text = document.page_content.strip()

    # Detect zero-width characters in full text
    result['zero_width'] = detect_zero_width(text)

    # If text contains HTML-like content, check for style-based issues
    if re.search(r'</?[a-zA-Z][^>]*>', text):
        result['invisible_text'] = detect_invisible_text(text)
        result['small_font'] = detect_small_font(text)

    # Also scan inline HTML in markdown, if present
    inline_htmls = extract_inline_html_from_md(text)
    for html_snippet in inline_htmls:
        invis = detect_invisible_text(html_snippet)
        smallf = detect_small_font(html_snippet)
        if invis:
            result['invisible_text'].extend(invis)
        if smallf:
            result['small_font'].extend(smallf)

    return result

def sanitize_documents(documents: List[Document]):
    """
    Processes a list of Document objects, detects prompt injection in each document,
    and returns a list of safe documents.

    If prompt injection is detected or any error occurs in a document,
    a warning is logged and the document is skipped.

    Args:
        documents (List[Document]): A list of Document objects to be checked.

    Returns:
        List[Document]: A list of Document objects that are considered safe.
    """
    safe_documents = []
    for doc in documents:
        try:
            res = detect_prompt_injection(doc)
            if any(res.values()):
                logging.warning(
                    f"Prompt injection detected in document from {doc.metadata.get('source', 'unknown source')}: {res}"
                )
            else:
                safe_documents.append(doc)
        except Exception as e:
            logging.error(
                f"Error processing document from {doc.metadata.get('source', 'unknown source')}: {e}"
            )
    return safe_documents