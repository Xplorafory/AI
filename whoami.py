import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
orgs = client.organization.list()

print("Active API Key (first 10 chars):", os.getenv("OPENAI_API_KEY")[:10] + "...")
print("Available orgs:", orgs)