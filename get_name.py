 
from transformers import AutoTokenizer, AutoModelForCausalLM

cache_dir = "/home/chrisvatt/data/huggingface_models"

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", trust_remote_code=True, cache_dir=cache_dir)
print(model)
