# LangChain Agent

This repo is paired with this [substack post](https://ilianherzi.substack.com/p/langchain-llama-jokester?r=219ah9&utm_campaign=post&utm_medium=web)

## Setup

1. Create a virtual environment and install dependencies based on requirments.txt
2. This project runs with python_dotenv, which means that you can easily keep track of your API keys by creating a `.env` file and saving them there. Or feel free to modify the API keys directly in `memory.py`. The format for the `env` file is:

```bash
MY_API_KEY=12345
MY_OTHER_KEY=54321
```

## Open AI token

1. Navigate to [OpenAI's platform](https://platform.openai.com/)
2. In the top right corner click on "Personal"
3. In the dropdown menu select "View API Keys"
4. Create a new secret key by clicking "Create new secret key" (ie: "my-secret-key").
5. Save your API key (preferrably in `.env`) as one you navigate away you won't be able to see it.

## Pinecone Vector Store DB token

1. Navigate to [pinecone.io](https://www.pinecone.io/)
2. Signup for free.
3. Once you sign up successfully you should see a landing page to fill in your info. Select python as your language of choice and an AI agent project.
4. Click create index.
5. Name your index. In `memory.py` this is your `PINECONE_INDEX_NAME`, which you can save in `.env`.

## Serp API token

1. Navigate to Serp API.
2. Fill in your info and sign up.
3. Replace SERPAPI_API_KEY with your key, again optionally saving in the `.env` file.

# Running the code.

Right now only running at desk is supported. This will change in the future.
Run with dotenv (make sure that `.env` is in the same directory youre running from). If we run from root then this is:

```bash
dotenv run python long_term_memory/memory.py --loop
```

OR

```bash
python long_term_memory/memory.py --loop
```

OR
The beta ipynb (not supported at this time)

```bash
jupyter notebook
# navigtate to Test.ipynb
```
