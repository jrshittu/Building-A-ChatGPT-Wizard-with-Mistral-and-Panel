# Building A ChatGPT Wizard with MistralAI and Panel

## Introduction
Mistral 7B is a super-smart language model with 7 billion parameters! It beats the best 13B model, Llama 2, in all tests and even outperforms the powerful 34B model, Llama 1, in reasoning, math, and code generation. How? Mistral 7B uses smart tricks like grouped-query attention (GQA) for quick thinking and sliding window attention (SWA) to handle all sorts of text lengths without slowing down.

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/mv9nkkrinccagi6z13ez.PNG)
Source: [Mistral.AI](https://mistral.ai/news/announcing-mistral-7b/)

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/qbclcim9okpeobmw4hs9.PNG)
Source: [Mistral.AI](https://mistral.ai/news/announcing-mistral-7b/)

And there's more! Mistral AI Team fine-tuned Mistral 7B for specific tasks with Mistral 7B – Instruct. It not only outshines Llama 2 13B in chat but also rocks both human and automated tests. Best part? Mistral 7B – was released under the Apache 2.0 license. 

## Using Mistral 7B Model
Mistral AI currently provides two types of access to Large Language Models: 
1. An API providing pay-as-you-go access to our latest models, the API key is not currently available for the general public but you can sign up on https://auth.mistral.ai/ui/registration to join the waitlist
2. Open source models available under the Apache 2.0 License, available on Hugging Face or directly from the documentation. This can be downloaded and used locally.

For this tutorial we'll use Mistral 7B Instruct v0.1 - GGUF

### About GGUF
GGUF is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML, which is no longer supported by llama.cpp.

List of some clients and libraries that are known to support GGUF:
1. ctransformers, a Python library with GPU accel, LangChain support, and OpenAI-compatible AI server.
2. llama-cpp-python, a Python library with GPU accel, LangChain support, and OpenAI-compatible API server.
3. text-generation-webui, the most widely used web UI, with many features and powerful extensions. Supports GPU acceleration.
4. LM Studio, an easy-to-use and powerful local GUI for Windows and macOS (Silicon), with GPU acceleration.
5. LoLLMS Web UI, a great web UI with many interesting and unique features, including a full model library for easy model selection.
6. Faraday.dev, an attractive and easy to use character-based chat GUI for Windows and macOS (both Silicon and Intel), with GPU acceleration.

## How to download GGUF files using ctransformers
Install ctransformers
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

Now, run the code below to download the models. Make sure to free up space on your computer and connect to a good internet connection.

```python
# imports the AutoModelForCausalLM class from the ctransformers library
from ctransformers import AutoModelForCausalLM

# load Mistral-7B-Instruct-v0.1-GGUF, Set gpu_layers to the number of layers to offload to GPU. The value is set to 0 because no GPU acceleration is available on my current system.

llm = AutoModelForCausalLM.from_pretrained("TheBloke/Mistral-7B-Instruct-v0.1-GGUF", model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf", model_type="mistral", gpu_layers=0)

# call the model to generate text, starting with the prompt "AI is going to"
print(llm("AI is going to"))
```

The output should look like this, 

![output real](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/r2zn5hsmrgl9srns4c9h.jpeg)

## Build a Mistral Chatbot with Panel
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



## Resources:

HuggingFace: [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

Panel: [https://panel.holoviz.org/](https://panel.holoviz.org/)


 
