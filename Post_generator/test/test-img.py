import os
import base64
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_API_KEY"),
    api_version="2024-02-01",
    azure_endpoint="https://gopal-moxqsanx-eastus2.cognitiveservices.azure.com"
)

result = client.images.generate(
    model="gpt-image-1-mini",   # deployment name
    prompt="A futuristic city at sunset",
    size="1024x1024"
)

image_base64 = result.data[0].b64_json

with open("output.png", "wb") as f:
    f.write(base64.b64decode(image_base64))

print("Image saved!")