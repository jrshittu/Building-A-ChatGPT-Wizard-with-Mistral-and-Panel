# Building A ChatGPT Wizard with MistralAI and Panel

## Introduction
Mistral 7B is a super-smart language model with 7 billion parameters! It beats the best 13B model, Llama 2, in all tests and even outperforms the powerful 34B model, Llama 1, in reasoning, math, and code generation. How? Mistral 7B uses smart tricks like grouped-query attention (GQA) for quick thinking and sliding window attention (SWA) to handle all sorts of text lengths without slowing down.

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/9dmy24ao6nnz66w5qc1x.PNG)

Source: [Mistral.AI Docs](https://mistral.ai/news/announcing-mistral-7b/)

And there's more! Mistral AI Team fine-tuned Mistral 7B for specific tasks with Mistral 7B – Instruct. It not only outshines Llama 2 13B in chat but also rocks both human and automated tests. Best part? Mistral 7B – was released under the Apache 2.0 license. 

In this article you'll learn about;

[Access to Mistral 7B Model](#use)

[Mistral 7B Instruct v0.1 - GGUF.](#7b)

[Building a Mistral Chatbot with Panel](#mistral)

[Building a Mistral Chatbot using API](#api)

[Adding memory to manage chat histories](#mem)

## Access to Mistral 7B Model <a name="use"></a>
Mistral AI currently provides two types of access to Large Language Models: 
1. An API providing pay-as-you-go access to the latest models, Sign up on https://auth.mistral.ai/ui/registration to join the waitlist.
2. Open source models available under the Apache 2.0 License, available on Hugging Face or directly from the documentation. 

## Mistral 7B Instruct v0.1 - GGUF. <a name="7b"></a>

GGUF is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML, which is no longer supported by llama.cpp.

List of some clients and libraries that are known to support GGUF:
1. ctransformers, a Python library with GPU accel, LangChain support, and OpenAI-compatible AI server.
2. llama-cpp-python, a Python library with GPU accel, LangChain support, and OpenAI-compatible API server.
3. text-generation-webui, the most widely used web UI, with many features and powerful extensions. Supports GPU acceleration.
4. LM Studio, an easy-to-use and powerful local GUI for Windows and macOS (Silicon), with GPU acceleration.
5. LoLLMS Web UI, a great web UI with many interesting and unique features, including a full model library for easy model selection.
6. Faraday.dev, an attractive and easy to use character-based chat GUI for Windows and macOS (both Silicon and Intel), with GPU acceleration.

### How to download GGUF files using ctransformers

**Install ctransformers**

```bash
# ctransformers with no GPU acceleration
pip install ctransformers

# ctransformers with CUDA GPU acceleration
pip install ctransformers[cuda]

# ctransformers with AMD ROCm GPU acceleration (Linux only)
CT_HIPBLAS=1 pip install ctransformers --no-binary ctransformers

# ctransformers with Metal GPU acceleration for macOS systems only
CT_METAL=1 pip install ctransformers --no-binary ctransformers
```

**Install Langchain**

```bash
pip install langchain
```

Now, run the code below to download and send a prompt to the model. Make sure to free up space on your computer and connect to a good internet connection.

```python
# import the AutoModelForCausalLM class from the ctransformers library
from ctransformers import AutoModelForCausalLM

# load Mistral-7B-Instruct-v0.1-GGUF, Set gpu_layers to the number of layers to offload to GPU. The value is set to 0 because no GPU acceleration is available on my current system.
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=0)

# call the model to generate text, starting with the prompt "AI is going to"
print(llm("AI is going to"))
```

The model will continue the statement as follows, 

![output real](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/r2zn5hsmrgl9srns4c9h.jpeg)

## Build a Mistral Chatbot with Panel<a name="mistral"></a>
### About Panel
Panel is an open-source Python library that lets you easily build powerful tools, dashboards and complex applications entirely in Python. To learn more about panel, kindly click [here](https://panel.holoviz.org/).

**Requirement:** Python 3.8 or later on Linux, Windows, and Mac.

**Installing Panel:** Open up a terminal and run the following command, which will install Panel with all its dependencies.

```bash
# If you're using pip
pip install panel

# if you're using conda
conda install panel
```
Now, let's demonstrates how to use the `ChatInterface` to create a chatbot using
[Mistral](https://docs.mistral.ai) through
[CTransformers](https://github.com/marella/ctransformers).

```python
#  imports the Panel library as pn.
import panel as pn

# import the AutoModelForCausalLM class from the ctransformers library
from ctransformers import AutoModelForCausalLM

# activate the Panel extension
pn.extension()

# Defines a callback function with three parameters: contents (the message content), user (the user sending the message), and instance (a pn.chat.ChatInterface instance).
async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    # Checks if a model named "mistral" is not in the llms dictionary. If not download and add it to the llms dictionary. 
    if "mistral" not in llms:
        instance.placeholder_text = "Let me download model, please wait..."
        llms["mistral"] = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            gpu_layers=0, # Set gpu_layers to the number of layers to offload to GPU. The value is set to 0 because no GPU acceleration is available on my current system.
        )

    llm = llms["mistral"]
    response = llm(contents, stream=True, max_new_tokens=1000)
    message = ""
    for token in response:
        message += token
        yield message

# Initialize an empty llms dictionary and create a pn.chat.ChatInterface instance to set the callback function for processing messages 
llms = {}
chat_interface = pn.chat.ChatInterface(callback=callback, callback_user="Mistral")
chat_interface.send(
    "Send a message to get a reply from Mistral AI!", user="System", respond=False
)

# Make the chat interface servable to a web server.
chat_interface.servable()
```

To launch a server using CLI and interact with this app, simply run `panel serve app.py`. Don't forget to save the script as `app.py`

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/8jesgxj65q39kmice8o8.PNG)

Click on http://localhost:5006/app to launch the Web UI in a browser and enter a prompt to chat with the model...

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/9t2nhkpni8o43a2kx0v2.gif)

## Build a Mistral Chatbot using API <a name="api"></a>

First install MistralAI
`pip install mistralai`

In order to use the Mistral API you'll need an API key. You can sign up for a Mistral account and create an API key from [here](https://auth.mistral.ai/ui/registration).

```python
import panel as pn
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

pn.extension()


async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    messages.append(ChatMessage(role="user", content=contents))

    mistral_response = ""
    for chunk in client.chat_stream(model="mistral-tiny", messages=messages):
        response = chunk.choices[0].delta.content
        if response is not None:
            mistral_response += response
            yield mistral_response

    if mistral_response:
        messages.append(ChatMessage(role="assistant", content=mistral_response))


messages = []
client = MistralClient()  # api_key=os.environ.get("MISTRAL_API_KEY", None)
chat_interface = pn.chat.ChatInterface(callback=callback, callback_user="Mistral AI")
chat_interface.send(
    "Send a message to get a reply from Mixtral!", user="System", respond=False
)
chat_interface.servable()
```

## Adding memory to manage chat histories<a name="mem"></a>
Let's Demonstrates how to use the `ChatInterface` to create a chatbot using
[Mistral](https://docs.mistral.ai) through
[CTransformers](https://github.com/marella/ctransformers). The chatbot includes a
memory of the conversation history.

```python

import panel as pn
from ctransformers import AutoConfig, AutoModelForCausalLM, Config

pn.extension()

SYSTEM_INSTRUCTIONS = "Do what the user requests."


def apply_template(history):
    history = [message for message in history if message.user != "System"]
    prompt = ""
    for i, message in enumerate(history):
        if i == 0:
            prompt += f"<s>[INST]{SYSTEM_INSTRUCTIONS} {message.object}[/INST]"
        else:
            if message.user == "Mistral":
                prompt += f"{message.object}</s>"
            else:
                prompt += f"""[INST]{message.object}[/INST]"""
    return prompt


async def callback(contents: str, user: str, instance: pn.chat.ChatInterface):
    if "mistral" not in llms:
        instance.placeholder_text = "Downloading model; please wait..."
        config = AutoConfig(
            config=Config(
                temperature=0.5, max_new_tokens=2048, context_length=2048, gpu_layers=1
            ),
        )
        llms["mistral"] = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            config=config,
        )

    llm = llms["mistral"]
    history = [message for message in instance.objects]
    prompt = apply_template(history)
    response = llm(prompt, stream=True)
    message = ""
    for token in response:
        message += token
        yield message


llms = {}
chat_interface = pn.chat.ChatInterface(
    callback=callback,
    callback_user="Mistral",
)
chat_interface.send(
    "Send a message to get a reply from Mistral!", user="System", respond=False
)
chat_interface.servable()
```

## Summary
Mistral 7B is a smart opensource model which demonstrates that language models may compress knowledge more than what was previously thought.

## Resources:

HuggingFace: [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

Panel: [https://panel.holoviz.org/](https://panel.holoviz.org/)

Github Repo: [Mistral Code Examples](https://github.com/holoviz-topics/panel-chat-examples/tree/main/docs/examples/mistral)


 
