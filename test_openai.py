import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("infra/.env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
resp = client.models.list()
print([m.id for m in resp.data][:5])