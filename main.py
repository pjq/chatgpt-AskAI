import argparse
import sys
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

    _, dir_basename = os.path.split(directory_path.rstrip(os.sep))
    json_filename = f"{dir_basename}.json"
    index.save_to_disk(json_filename)

    return index


def ask_ai(directory_path):
    _, dir_basename = os.path.split(directory_path.rstrip(os.sep))
    json_filename = f"{dir_basename}.json"
    index = GPTSimpleVectorIndex.load_from_disk(json_filename)
    while True:
        # query = input("Ask: \n")
        sys.stdout.write("Ask: \n")
        sys.stdout.flush()
        query = sys.stdin.readline().strip()
        response = index.query(query, response_mode="compact")
        print(f"Answer: {response.response}\n")
        # display(Markdown(f"Response: <b>{response.response}</b>"))

def batch_ask(directory_path, comments_file):
    json_filename = f"{os.path.basename(directory_path)}.json"
    index = GPTSimpleVectorIndex.load_from_disk(json_filename)

    with open(comments_file, 'r') as file:
        comment_lines = file.readlines()

    for line in comment_lines:
        if line.startswith("userName:"):
            comment = line.split("content: ")[1].strip()
            response = index.query(comment, response_mode="compact")
            print(f"Comments: {comment}")
            print(f"AutoReply: {response.response}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory_path",
        dest="directory_path",
        type=str,
        default="data",
        help="Path to the directory that contains the data",
    )

    parser.add_argument(
        "--comments",
        dest="comments",
        type=str,
        help="Path to the comments.txt file",
    )

    options = parser.parse_args()

    http_proxy = os.environ.get("http_proxy")
    https_proxy = os.environ.get("https_proxy")
    print(f"http_proxy: {http_proxy}, https_proxy: {https_proxy}")
    if http_proxy:
        os.environ["HTTP_PROXY"] = http_proxy
    if https_proxy:
        os.environ["HTTPS_PROXY"] = https_proxy

    _, dir_basename = os.path.split(options.directory_path.rstrip(os.sep))
    json_filename = f"{dir_basename}.json"
    if not os.path.exists(json_filename):
        construct_index(options.directory_path)

    print(f"data path: {options.directory_path}, index json: {json_filename},  comments: {options.comments}")

    if options.comments:
        batch_ask(options.directory_path, options.comments)
    else:
        ask_ai(options.directory_path)
