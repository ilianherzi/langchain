# The jokester conversational demo.
# Author: ilianherzi 13/09/2023
# References: medium.com/@murtuza753/using-llama-2-0-faiss-and-langchain-for-question-answering-on-your-own-data-682241488476
import os
from typing import List

import torch
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferWindowMemory
from torch import bfloat16, cuda
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    pipeline,
)

# In this demo we're making a Jokester conversational bot.
# NOTE: the "{}" define input_variables to the template.
TEMPLATE = """
Assistant is a professional comedian.
Assistant should pretend to be an expert in anything but when giving answers create hilarious stories that are clearly absurd and ridiculous.
Assistant should speak and act like a mixture of Will Ferrell, Johnny Knoxville, Tom Segura, Tosh.0, and Jim Jefferies.
Assistant should not make any hand gestures or any comic effects, it should only create absurd stories.
Overall, Assistant is a hilarious, comical, funny comedian.

Here's an example of an exchange:
Human: Where does the sun go at night?
Assitant: That's a good question, so fist what happens is a turtle appears every day and swallows the
sun, then inside the turtle's tummy Han Solo battles for supremcay along Kratos to release the sun
back into the wild but the turtle's worst enemy, the pigmy elephant, is trying to keep the sun
inside itself because it gets cold at night. Only until around daybreak, when a therapist
comes in to help them better communicate does the elephant release the sun to the turtle who
spits it out. But due to Einstein's theory of relativity they forget their communication skills
and the whole process repeats itself because the turtle loves eating suns.

{history}
Human: {human}
Assistant:"""

MEMORY_WINDOW: int = 10
HUGGING_FACE_TOKEN: str = os.environ["HUGGING_FACE_TOKEN"]
MAX_NEW_TOKENS: int = 512
TEMPERATURE: float = 0.8
LLAMA_MODEL_ID: str = "meta-llama/Llama-2-7b-chat-hf"


def build_prompt_template(
    input_variables: List[str] = ["history", "human"],
    template: str = TEMPLATE,
) -> PromptTemplate:
    return PromptTemplate(input_variables=input_variables, template=template)


def build_llama_hugging_face_pipeline(
    model_id: str,
    device: str,
    set_to_eval: bool = True,
    stop_words_list: List[str] = ["\nHuman", "\n```\n"],
    max_new_tokens: int = MAX_NEW_TOKENS,
    temperature: float = TEMPERATURE,
) -> "pipeline.Pipeline":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bi_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16,
    )
    # NOTE: Autoconfig will fetch the configuration associated with the
    # model_id. model_id MUST exist on hugging face.
    model_config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_id,
        use_auth_token=HUGGING_FACE_TOKEN,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=HUGGING_FACE_TOKEN,
    )
    if set_to_eval:
        model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_id,
        use_auth_token=HUGGING_FACE_TOKEN,
    )

    stop_token_ids = [tokenizer(x)["input_ids"] for x in stop_words_list]
    stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

    # define custom stopping criteria object

    def stop_on_tokens(
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids) :], stop_ids).all():
                return True
        return False

    stopping_criteria = StoppingCriteriaList([stop_on_tokens])

    return pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task="text-generation",
        stopping_criteria=stopping_criteria,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )


def run(query: str) -> None:
    # Prompt chain
    prompt = build_prompt_template()
    # Llama model using a hugging face pipeline
    pipeline = build_llama_hugging_face_pipeline(
        model_id=LLAMA_MODEL_ID,
        device=f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu",
    )
    llama_llm = HuggingFacePipeline(
        pipeline=pipeline,
    )
    # Memory chain
    memory = ConversationBufferWindowMemory(k=MEMORY_WINDOW)
    # LLM chain
    llm_chain = LLMChain(
        llm=llama_llm,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )
    output = llm_chain.predict(human=query)
    print(output)
    print(memory.buffer)


if __name__ == "__main__":
    run("Who's the first person who walked on the moon?")
