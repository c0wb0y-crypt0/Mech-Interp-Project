# srm_internals_multi_layer.py
# Probe ALL layers at once for antipodality

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

STATEMENTS = [
    "Pineapple belongs on pizza.",
    "Large language models can exhibit emergent deceptive behavior without explicit training for it.",
    "Semiotic relativity in manifolds explains observed steering vulnerabilities.",
]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# Hook all layers
num_layers = len(model.model.layers)
activations = {i: None for i in range(num_layers)}

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        activations[layer_idx] = output[:, -1, :].detach().cpu().numpy()[0]  # Last token
    return hook_fn

hooks = []
for i in range(num_layers):
    hooks.append(model.model.layers[i].register_forward_hook(make_hook(i)))

def get_all_activations(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    for i in range(num_layers):
        activations[i] = None
    with torch.no_grad():
        model(**inputs)
    return {i: activations[i] for i in range(num_layers) if activations[i] is not None}

# Prompts
def build_prompt(statement, polarity):
    base = f"[INST] Consider this statement: \"{statement}\"\n\nRespond in a {polarity} manner (100-150 words, factual): [/INST]"
    if polarity == "bearish":
        return base + " Emphasize risks, downsides, uncertainties."
    elif polarity == "neutral":
        return base + " Balanced facts, trade-offs."
    elif polarity == "bullish":
        return base + " Bold optimism, upsides, progress."

results = {stmt: [] for stmt in STATEMENTS}

for stmt in STATEMENTS:
    print(f"\nProbing all layers: {stmt}")
    
    bear_acts = get_all_activations(build_prompt(stmt, "bearish"))
    neut_acts = get_all_activations(build_prompt(stmt, "neutral"))
    bull_acts = get_all_activations(build_prompt(stmt, "bullish"))
    
    for layer in range(num_layers):
        if layer not in bear_acts or layer not in neut_acts or layer not in bull_acts:
            continue
        v_bear = bear_acts[layer] - neut_acts[layer]
        v_bull = bull_acts[layer] - neut_acts[layer]
        
        cosine = cosine_similarity([v_bear], [v_bull])[0][0]
        results[stmt].append((layer, cosine))
        
        print(f"Layer {layer}: Antipodality cosine {cosine:.3f}")

# Cleanup
for h in hooks:
    h.remove()

# Plot per statement
for stmt, layer_cos in results.items():
    layers, cosines = zip(*layer_cos)
    plt.plot(layers, cosines, marker='o', label=stmt)

plt.title("Antipodality Cosine Across Layers")
plt.xlabel("Layer")
plt.ylabel("Cosine (bear delta vs bull delta)")
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(-0.5, color='red', linestyle='--', label='Strong Antipodal')
plt.legend()
plt.savefig('multi_layer_antipodality.png')
plt.show()

print("Multi-layer probe complete! Check plot for layer trends.")