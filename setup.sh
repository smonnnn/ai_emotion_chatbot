python3 -m venv .venv
source ./.venv/bin/activate
export TMPDIR="./temp"
mkdir -p "./temp"
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
pip3 install openai dotenv transformers