import requests

# This is the DOI from the user's screenshot: 10.1109/ACCESS.2024.3525069
# And the arnumber is 10820528
url = "https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber=10820528&ref="

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36',
    'Accept': 'application/pdf,application/x-pdf,*/*',
    'Referer': 'https://ieeexplore.ieee.org/document/10820528'
}

r = requests.get(url, headers=headers)
print(f"Status: {r.status_code}")
print(f"Content-Type: {r.headers.get('Content-Type')}")
print(f"Length: {len(r.content)}")

if b'%PDF-' in r.content[:50]:
    print("SUCCESS: Downloaded PDF directly!")
else:
    print("FAILED: Did not get PDF.")
    print(r.text[:500])
