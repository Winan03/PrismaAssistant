"""Inspeccionar HTML crudo de sci-hub.box para entender la estructura del DOM."""
import requests
import re

url = 'https://sci-hub.box/10.1109/ACCESS.2021.3095559'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,*/*',
}

print(f"Fetching: {url}")
r = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('Content-Type', '')}")
print(f"Content-Length: {len(r.content)}")
print(f"Final URL: {r.url}")

html = r.text

# Check if it's a PDF directly
if r.content[:5] == b'%PDF-':
    print("\n*** RESPONSE IS A DIRECT PDF! ***")
else:
    print(f"\n--- RAW HTML (first 8000 chars) ---")
    print(html[:8000])
    print("--- END ---")

# Search for patterns
patterns = {
    'iframe': r'<iframe[^>]*>',
    'embed': r'<embed[^>]*>',
    'object': r'<object[^>]*>',
    'pdf_in_href': r'href="[^"]*\.pdf[^"]*"',
    'pdf_in_src': r'src="[^"]*\.pdf[^"]*"',
    'any_src': r'src="[^"]*"',
    'any_href': r'href="[^"]*"',
    'onclick': r'onclick="[^"]*"',
    'button': r'<button[^>]*>[^<]*</button>',
    'location_href': r'location\.href\s*=\s*["\'][^"\']*["\']',
    'window_open': r'window\.open\s*\([^)]*\)',
    'download': r'download',
    'save_btn': r'id="save"[^>]*',
    'pdf_id': r'id="pdf"[^>]*',
}

print("\n--- PATTERN SEARCH ---")
for name, pat in patterns.items():
    matches = re.findall(pat, html, re.IGNORECASE)
    if matches:
        print(f"\n[{name}] ({len(matches)} matches):")
        for m in matches[:10]:
            print(f"  -> {m[:200]}")
