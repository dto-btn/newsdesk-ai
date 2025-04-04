import os
import sys

import pymupdf4llm
import tiktoken
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
)

azure_openai_uri = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = os.getenv("AZURE_OPENAI_VERSION", "2024-05-01-preview")

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=str(azure_openai_uri),
    azure_ad_token_provider=token_provider,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file>")
        return

    input_file = "./documents/" + sys.argv[1]
    print(f"Processing file: {input_file}")
    md_text = pymupdf4llm.to_markdown(input_file, write_images=False)
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = num_tokens_from_string(md_text, encoding.name)
    print(f"Number of tokens: {tokens}")
    sentimemnt_analysis(md_text)


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def sentimemnt_analysis(document: str):
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You will do sentiment analysis on the document provided. They are clippings of news articles that pertain to the SSC (Shared Services Canada) media team.",
            },
            {
                "role": "user",
                "content": document,
            },
        ],
        max_tokens=4096,
        temperature=1.0,
        top_p=1.0,
        model="gpt-4o",
    )

    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
