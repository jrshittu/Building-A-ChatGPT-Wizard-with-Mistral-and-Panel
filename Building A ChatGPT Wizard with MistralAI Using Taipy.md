# Building A ChatGPT Wizard with MistralAI Using Taipy

## Introduction
Mistral 7B is a super-smart language model with 7 billion parameters! It beats the best 13B model, Llama 2, in all tests and even outperforms the powerful 34B model, Llama 1, in reasoning, math, and code generation. How? Mistral 7B uses smart tricks like grouped-query attention (GQA) for quick thinking and sliding window attention (SWA) to handle all sorts of text lengths without slowing down.

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/9dmy24ao6nnz66w5qc1x.PNG)

Source: [Mistral.AI Docs](https://mistral.ai/news/announcing-mistral-7b/)

And there's more! Mistral AI Team fine-tuned Mistral 7B for specific tasks with Mistral 7B – Instruct. It not only outshines Llama 2 13B in chat but also rocks both human and automated tests. Best part? Mistral 7B – was released under the Apache 2.0 license. 

In this article you'll learn about;

[Access to Mistral 7B Model](#use)

[Mistral 7B Instruct v0.1 - GGUF.](#7b)

[Building a Mistral Chatbot with Taipy](#mistral)

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

## Build a Mistral Chatbot with Taipy<a name="mistral"></a>
### About Taipy

Taipy is a Python open-source library that makes it simple to create data-driven web applications. It takes care of both the visible part(Frontend) and the behind-the-scenes(Backend) operations. Its goal is to speed up the process of developing applications, from the early design stages to having a fully functional product ready for use.

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/k24u6ko4tkjffice6thz.gif)

Source: [Taipy Docs](https://docs.taipy.io/en/latest/)

**Requirement:** Python 3.8 or later on Linux, Windows, and Mac. 

**Installing Taipy:** Open up a terminal and run the following command, which will install Taipy with all its dependencies.

```bash
pip install taipy
```
_If you run into any trouble installing the library, you can try a more stable python version like v3.11_

We're set, let say hello to Taipy...

```python
# import the library
from taipy import Gui

hello = "# Hello Taipy!" 

# run the gui
Gui(hello).run()
```

Save the code as a Python file: e.g., `hi_taipy.py`. 
Run the code and wait for the client link `http://127.0.0.1:5000` to display and pop up in your browser. 
You can change the port if you want to run multiple servers at the same time with `Gui(...).run(port=xxxx)`.

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/6w2h2jryumg8kumid3ms.PNG)

Now, let's demonstrates how to use the use [Taipy](https://docs.taipy.io/en/latest/getting_started/) to create a chatbot using
[Mistral](https://docs.mistral.ai) through
[CTransformers](https://github.com/marella/ctransformers).

### Step 1: Create Chat layout with Taipy

```python
# import Gui to create and manage graphical user interfaces.
from taipy.gui import Gui

# Define Taipy chat layout, add a table element that occupies the full width of the available space
# Create an input field with a label, using a state variable named current_prompt to store its value.
# Create a send button

chat = """
<|table|show_all|width=100%|>
<|{current_prompt}|input|label=Enter a prompt here...|class_name=fullwidth|>
<|button|label=Send|> 
"""

# Instantiate a Gui object with the defined layout and starts the UI event loop, render and display the interface.
Gui(chat).run()
```

Now save and run the App

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/86jmtz6yj07f30h4cw15.PNG)

### Step 2: Add Sidebar
Now let's create a two-column layout.

```python
from taipy.gui import Gui

# Create a two-column layout with a fixed 300px width for the first column.
chat = """
<|layout|columns=300px 1|
<|part|render=True|class_name=sidebar|
# Taipy **Chat**{: .color-primary} # {: .logo-text}

<|Chat History|button|class_name=fullwidth plain|on_action=previous_chat|>
|>

<|part|render=True|class_name=p2 align-item-bottom table|
<|{conversation}|table|style=style_conv|show_all|width=100%|selected={selected_row}|rebuild|>
<|part|class_name=card mt1|
<|{current_user_message}|input|label=Enter a prompt here...|on_action=send_message|class_name=fullwidth|>
<|Send Prompt|button|class_name=plain fullwidth|on_action=send_message|>
|>
|>
|>
"""

# Instantiate a Gui object with the defined layout and starts the UI event loop, render and display the interface in light mode.
Gui(chat).run(dark_mode=False)
```
Run the code.....

![Image description](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/2gmxrl793irrspj1qvgt.PNG)


```python
import taipy as tp
from ctransformers import AutoModelForCausalLM

# LLM loading and response generation function
@tp.func
def generate_response(message: str) -> str:
    """Loads the Mistral LLM and generates a response to a given message."""

    # Load the LLM lazily (only when needed)
    llm = tp.lazy(
        lambda: AutoModelForCausalLM.from_pretrained(
            "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
            model_file="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            gpu_layers=0,  # Adjust GPU usage if needed
        )
    )

    # Generate response tokens
    response = llm()(message, stream=True, max_new_tokens=1000)

    # Construct response string
    response_message = ""
    for token in response:
        response_message += token

    return response_message

# Define the layout elements
chat_messages = tp.textarea(label="Chat Messages", rows=10, disabled=True)
user_message = tp.text_input(label="Your Message")
send_button = tp.button("Send")

# Arrange elements vertically
layout = tp.vbox(chat_messages, user_message, send_button)

# Callback function for the send button
@send_button.on_click
def handle_message_send():
    user_message_text = user_message.value  # Retrieve user message
    response_text = generate_response(user_message_text)  # Generate LLM response

    # Update chat messages
    chat_messages.value += f"You: {user_message_text}\n"
    chat_messages.value += f"Mistral: {response_text}\n"

    # Clear the input field
    user_message.value = ""

# Display the layout
tp.show(layout)
```
`To run this code:`

Save it as a Python file (e.g., `chat_app.py`)

Run the app: `taipy run chat_app.py`


## Build a Mistral Chatbot using API <a name="api"></a>

First install MistralAI
`pip install mistralai`

In order to use the Mistral API you'll need an API key. You can sign up for a Mistral account and create an API key from [here](https://auth.mistral.ai/ui/registration).


## Adding memory to manage chat histories<a name="mem"></a>

## Summary
Mistral 7B is a smart opensource model which demonstrates that language models may compress knowledge more than what was previously thought.

## Resources:

HuggingFace: [Mistral-7B-Instruct-v0.1-GGUF](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)

Taipy: [Taipy Docs](https://docs.taipy.io/en/latest/)

 