
## Getting Started 

To run this code, you need first install `uv`. If not, you can install use the following command:

```Shell
wget -qO- https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv --version
```

After that, clone this repository and install the dependencies using `uv`:

```Shell
git clone https://github.com/YYZhang2025/nano-VLM.git
cd nano-VLM

uv venv 
source .venv/bin/activate
uv sync
```

To train the code, make sure you have `.env` file in the root directory, and has following keys:
```Text
WANDB_API_KEY=
WANDB_ENTITY=
WANDB_PROJECT=
```
to enable the logging on the [Wandb](https://wandb.ai/site/).
