import pymupdf4llm
import tiktoken
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <input_file>")
        return

    input_file =  "./documents/" + sys.argv[1]
    print(f"Processing file: {input_file}")
    md_text = pymupdf4llm.to_markdown(input_file, write_images=False)
    encoding = tiktoken.encoding_for_model('gpt-4o')
    tokens = num_tokens_from_string(md_text, encoding.name)
    print(f"Number of tokens: {tokens}")
    # now work with the markdown text, e.g. store as a UTF8-encoded file
    #import pathlib
    #pathlib.Path("output.md").write_bytes(md_text.encode())

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

if __name__ == "__main__":
    main()
