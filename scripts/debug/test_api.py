import requests
import json

title = "Applications of Machine Learning and Deep Learning in Antenna Design, Optimization, and Selection: A Review"
url = f"https://www.scienceopen.com/api/search"
# Let's see if we can find the API endpoint from web search or just try to guess
print("Can't easily guess ScienceOpen API. Let's try Crossref by title instead, it's a standard open API!")

url_cr = f"https://api.crossref.org/works?query.title={requests.utils.quote(title)}&select=DOI,URL,link&rows=1"
r = requests.get(url_cr)
print("Crossref:", r.json())
