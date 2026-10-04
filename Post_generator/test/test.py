import os
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

endpoint = "https://devops-testing-2026.cognitiveservices.azure.com/"
model_name = "gpt-5.4-nano"
deployment = "gpt-5.4-nano"

subscription_key = os.getenv("API_KEY")
api_version = "2024-12-01-preview"

client = AzureOpenAI(
    api_key=subscription_key,
    azure_endpoint=endpoint,
    api_version=api_version
)

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "say hello in one sentence",
        }
    ],
    max_completion_tokens=16384,
    model=deployment
)

print(response.choices[0].message.content)