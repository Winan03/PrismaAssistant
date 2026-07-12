import requests

title = "Applications of Machine Learning and Deep Learning in Antenna Design, Optimization, and Selection: A Review"
url = f"https://sci-hub.se/{title}"
headers = {'User-Agent': 'Mozilla/5.0'}
r = requests.get(url, headers=headers)
print(f"Status Code: {r.status_code}")
if 'application/pdf' in r.text or '<object' in r.text or '<iframe' in r.text:
    print("Found PDF embed!")
else:
    print("PDF embed not found.")

# Let's try POST
r2 = requests.post("https://sci-hub.se/", data={"request": title}, headers=headers)
print(f"POST Status Code: {r2.status_code}")
if 'application/pdf' in r2.text or '<object' in r2.text or '<iframe' in r2.text:
    print("POST Found PDF embed!")
else:
    print("POST PDF embed not found.")
