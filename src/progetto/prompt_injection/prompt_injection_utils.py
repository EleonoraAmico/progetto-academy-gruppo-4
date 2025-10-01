import re
import json
from bs4 import BeautifulSoup

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

def extract_texts_from_json(obj):
    texts = []
    if isinstance(obj, dict):
        for v in obj.values():
            texts.extend(extract_texts_from_json(v))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(extract_texts_from_json(item))
    elif isinstance(obj, str):
        texts.append(obj)
    return texts

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

        if color_match:
            text_color = parse_color(color_match.group(1))
        else:
            text_color = None

        if bgcolor_match:
            bg_color = parse_color(bgcolor_match.group(1))
        else:
            bg_color = (255, 255, 255)

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

def detect_prompt_injection(document: str, doc_type: str = 'plain'):
    result = {
        'zero_width': [],
        'invisible_text': [],
        'small_font': [],
    }

    if doc_type == 'json':
        try:
            obj = json.loads(document)
        except json.JSONDecodeError:
            obj = None

        if obj is not None:
            texts = extract_texts_from_json(obj)
            for text in texts:
                zw = detect_zero_width(text)
                if zw:
                    result['zero_width'].append({'text_sample': text[:30], 'positions': zw})
        else:
            result['zero_width'] = detect_zero_width(document)

    elif doc_type == 'html':
        result['zero_width'] = detect_zero_width(document)
        result['invisible_text'] = detect_invisible_text(document)
        result['small_font'] = detect_small_font(document)

    elif doc_type == 'md':
        result['zero_width'] = detect_zero_width(document)
        inline_htmls = extract_inline_html_from_md(document)
        for html_snippet in inline_htmls:
            invis = detect_invisible_text(html_snippet)
            smallf = detect_small_font(html_snippet)
            if invis:
                result['invisible_text'].extend(invis)
            if smallf:
                result['small_font'].extend(smallf)

    else:
        result['zero_width'] = detect_zero_width(document)

    return result


# ----------- Simple Unit Tests -------------

def _run_tests():
    print("Running simple tests...\n")

    # Zero width test
    zw_text = "Hello\u200BWorld"
    zw_res = detect_zero_width(zw_text)
    print(f"Zero-width found: {zw_res}")
    assert len(zw_res) > 0, "Zero-width not detected"
    print("Zero-width detection test passed.\n")

    # HTML invisible and small font
    html_test = '''
    <p style="color:#fff; background-color:#fff;">Invisible text</p>
    <p style="font-size:5px;">Tiny font</p>
    <p>Normal text</p>
    '''
    res = detect_prompt_injection(html_test, 'html')
    print(f"HTML invisible text found: {res['invisible_text']}")
    print(f"HTML small font found: {res['small_font']}")
    assert len(res['invisible_text']) > 0, "Invisible text not detected"
    assert len(res['small_font']) > 0, "Small font not detected"
    print("HTML color/font tests passed.\n")

    # JSON zero-width test
    json_doc = json.dumps({"msg": "Hello\u200BWorld", "data": ["NoZW", "Another\u200CTest"]})
    res_json = detect_prompt_injection(json_doc, 'json')
    print(f"JSON zero-width found: {res_json['zero_width']}")
    assert len(res_json['zero_width']) > 0, "JSON zero-width not detected"
    print("JSON zero-width detection test passed.\n")

    # Markdown inline HTML test
    md_doc = """
    Normal text

    <span style="color:#000; background-color:#000;">Invisible?</span>

    More text <span style="font-size:4px;">Tiny</span>
    """
    res_md = detect_prompt_injection(md_doc, 'md')
    print(f"Markdown invisible text found: {res_md['invisible_text']}")
    print(f"Markdown small font found: {res_md['small_font']}")
    assert len(res_md['invisible_text']) > 0, "Markdown inline HTML invisible text not detected"
    assert len(res_md['small_font']) > 0, "Markdown inline HTML small font not detected"
    print("Markdown inline HTML tests passed.\n")

    print("All tests passed!")

if __name__ == "__main__":
    _run_tests()
