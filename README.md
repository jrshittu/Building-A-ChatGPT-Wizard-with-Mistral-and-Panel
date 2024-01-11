# Building A ChatGPT Wizard with MistralAI and Panel

![Gif](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/m07pkz4y22fmslfzqs5o.gif)

## Introduction
Mistral 7B is a super-smart language model with 7 billion parameters! It beats the best 13B model, Llama 2, in all tests and even outperforms the powerful 34B model, Llama 1, in reasoning, math, and code generation. How? Mistral 7B uses smart tricks like grouped-query attention (GQA) for quick thinking and sliding window attention (SWA) to handle all sorts of text lengths without slowing down.

And there's more! Mistral AI Team fine-tuned Mistral 7B for specific tasks with Mistral 7B – Instruct. It not only outshines Llama 2 13B in chat but also rocks both human and automated tests. Best part? Mistral 7B – was released under the Apache 2.0 license. 

## Using Mistral 7B Model
Mistral AI currently provides two types of access to Large Language Models: 
1. An API providing pay-as-you-go access to our latest models, the API key is not currently available for the general public but you can sign up on https://auth.mistral.ai/ui/registration to join the waitlist
2. Open source models available under the Apache 2.0 License, available on Hugging Face or directly from the documentation. This can be downloaded and used locally.

For this tutorial we'll use Mistral 7B Instruct v0.1 - GGUF.

### About GGUF
GGUF is a new format introduced by the llama.cpp team on August 21st 2023. It is a replacement for GGML, which is no longer supported by llama.cpp.

List of some clients and libraries that are known to support GGUF:
1. ctransformers, a Python library with GPU accel, LangChain support, and OpenAI-compatible AI server.
2. llama-cpp-python, a Python library with GPU accel, LangChain support, and OpenAI-compatible API server.
3. llama.cpp. The source project for GGUF. Offers a CLI and a server option.
4. text-generation-webui, the most widely used web UI, with many features and powerful extensions. Supports GPU acceleration.
5. KoboldCpp, a fully featured web UI, with GPU accel across all platforms and GPU architectures. Especially good for story telling.
6. LM Studio, an easy-to-use and powerful local GUI for Windows and macOS (Silicon), with GPU acceleration.
7. LoLLMS Web UI, a great web UI with many interesting and unique features, including a full model library for easy model selection.
8. Faraday.dev, an attractive and easy to use character-based chat GUI for Windows and macOS (both Silicon and Intel), with GPU acceleration.

## How to download GGUF files using ctransformers
Install the package
```bash
# Base ctransformers with no GPU acceleration
pip install ctransformers

# Or with CUDA GPU acceleration
pip install ctransformers[cuda]

# With AMD ROCm GPU acceleration (Linux only)
CT_HIPBLAS=1 pip install ctransformers --no-binary ctransformers

# With Metal GPU acceleration for macOS systems only
CT_METAL=1 pip install ctransformers --no-binary ctransformers
```

 