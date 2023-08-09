# llm-demo
An exploration of how a general-purpose Large Language Model (LLM) might be deployed in a Canadian federal context.

Note: this code is intended as a starting point for developers wanting to explore the application space. The code is intended to be read as a learning experience.

Key considerations before deploying this to anything resembling real users:
* The memory and CPU requirements for this are not tiny: You need at least 16GB of memory and 4 relatively quick real CPU cores (not hyper-thread cores) to run this realistically.
* This repo does not address key concerns like authentication or encryption in transit. You can achieve these fairly easily using something like https://github.com/oauth2-proxy/oauth2-proxy and a TLS terminator like NGINX.
* If the service gets overloaded, it *will* crash and dump core. The service has no persistent state, so you can make it recover using a shell script, a system service, a container orchestration solution, or whatever floats your boat. 
* While this will technically run on processors without AVX2 extensions, it's really not recommended.

# Quickstart:
## Installation
```
wget https://huggingface.co/TheBloke/airoboros-l2-7B-gpt4-m2.0-GGML/resolve/main/airoboros-l2-7b-gpt4-m2.0.ggmlv3.q4_0.bin
pip install -r requirements.txt
```
## Execution
```
python app.py
```

# Getting started with LLM learning
Open `app.py` in your favourite editor, and follow along if you want to learn about interacting with LLMs.
If you're interested in the websocket part of the interaction, take a look at `templates/index.html`. It is similarly not intended as an example of best practice, but as a learning experience.
