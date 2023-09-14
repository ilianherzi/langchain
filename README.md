# langchain

This repo is paired with this [substack post](https://ilianherzi.substack.com/p/langchain-llama-jokester?r=219ah9&utm_campaign=post&utm_medium=web)

## Access to Llama 2 model from Meta

1. Make a hugging face account with your@email.here. Ensure you have the correct email at [hugging settings](https://huggingface.co/settings/account).
2. Using the _**same email**_ as your hugging face account request access to [Llama-2](https://ai.meta.com/llama/) from meta.
3. For your [hugging settings](https://huggingface.co/settings/account) get a hugging face token and set `HUGGING_FACE_TOKEN` in `demo.py`.

## Running the demo.

1. Navigate to [Google colab](https://colab.research.google.com/).
2. Upload the notebook found in this repo `demo.ipynb`.
3. Run the cells and modify `query` to what you want

OR

1. Run at desk using

```bash
HUGGING_FACE_TOKEN=your-token demo.py
```
