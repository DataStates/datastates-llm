# datastates-llm
LLM checkpointing engine for DeepSpeed/Megatron

### Install and test
```
git clone https://github.com/DataStates/datastates-llm.git
cd datastates-llm/
git checkout dev            # Checkout to dev branch for now.
pip install . -v            # Installs the CPP/Python binding.
python test_datastates.py   # Test with a simple PyTorch code.
```