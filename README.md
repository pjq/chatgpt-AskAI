# AI Question Answering System
This is a simple AI Question Answering System based on the OpenAI GPT3. The system requires a directory that contains the data to be used to generate the index.

## Requirements
- langchain
- gpt-index
- OpenAI API key

```shell
pip3 install -r requirements.txt
```
## Usage
Create the index:
```shell

python3 main.py --directory_path data
```

## Arguments
--directory_path: Path to the directory that contains the data (default: "data")

## Environment Variables
OPENAI_API_KEY: OpenAI API key

## License
This project is licensed under the MIT License.


## Reference
- https://uxdesign.cc/i-built-an-ai-that-answers-questions-based-on-my-user-research-data-7207b052e21c
- https://colab.research.google.com/drive/1PQXcM_jhN6QJ7uTkxvNbxoI54r03uSr3?usp=sharing#scrollTo=RoJHE4fsAT3w