import argparse
import os
from gpt_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from IPython.display import Markdown, display


def construct_index(directory_path):
    print(f"construct_index: {directory_path}")
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 300
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.5, model_name="text-davinci-003", max_tokens=num_outputs, openai_api_key=openai_api_key))
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = GPTSimpleVectorIndex(
        documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    )

    index.save_to_disk('index.json')

    return index


def ask_ai():
    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    while True:
        query = input("What do you want to ask? ")
        response = index.query(query, response_mode="compact")
        print(response.response)
        # display(Markdown(f"Response: <b>{response.response}</b>"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory_path",
        dest="directory_path",
        type=str,
        default="data",
        help="Path to the directory that contains the data",
    )
    options = parser.parse_args()

    http_proxy = os.environ.get("http_proxy")
    https_proxy = os.environ.get("https_proxy")
    print(f"http_proxy: {http_proxy}, https_proxy: {https_proxy}")
    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy

    if not os.path.exists("index.json"):
        construct_index(options.directory_path)
    ask_ai()
