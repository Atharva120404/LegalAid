import requests
from pprint import pprint

url = "http://localhost:5000/score_query"
payload = {
    "query": "What is the legal minimum drinking age in India?",
    "user_context": {}
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
pprint(response.json()) 