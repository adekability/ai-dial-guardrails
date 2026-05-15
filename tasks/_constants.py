import os

from langchain_openai import AzureChatOpenAI
from pydantic import SecretStr

DIAL_URL = 'https://ai-proxy.lab.epam.com'
API_KEY = os.getenv('DIAL_API_KEY', '')
MODEL = 'gpt-4.1-nano-2025-04-14'
API_VERSION = '2024-08-01-preview'


def create_llm(temperature: float = 0) -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=DIAL_URL,
        api_key=SecretStr(API_KEY) if API_KEY else None,
        api_version=API_VERSION,
        azure_deployment=MODEL,
        model=MODEL,
        temperature=temperature,
    )