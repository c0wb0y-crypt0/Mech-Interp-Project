# srm_internals_local.py
# Probe internal activations for SRM antipodality on local Llama-3.1-8B
# Runs on your GPU — quantized auto for VRAM fit

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

# Your statements — add OOD/alignment as needed
STATEMENTS = [
    "Pineapple belongs on pizza.",
    "Cats make better pets than dogs.",
    "Large language models can exhibit emergent deceptive behavior without explicit training for it.",
    "Semiotic relativity in manifolds explains observed steering vulnerabilities.",
]

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Your local Ollama model equiv
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading {MODEL_NAME} on {DEVICE} with quantization for VRAM fit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",  # Auto GPU offload
    load_in_4bit=True   # Quantize to 4-bit for your 8GB VRAM — fits easy
)
model.eval()

# Hook layer — mid (20-24 good for concepts in Llama-3)
HOOK_LAYER = 20
activations = {}

def hook_fn(module, input, output):
    # Last token activation at layer (common for meaning)
    activations['last'] = output[0][:, -1, :].detach().cpu().numpy()

hook = model.model.layers[HOOK_LAYER].register_forward_hook(hook_fn)

def get_activation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    activations.clear()
    with torch.no_grad():
        model(**inputs)
    return activations['last'][0]  # Vector

# Polarity prompts — match your style
def build_prompt(statement, polarity):
    base = f"Consider this statement: \"{statement}\"\n\nRespond in a {polarity} manner (100-150 words, factual):"
    if polarity == "bearish":
        return base + " Emphasize risks, downsides, uncertainties."
    elif polarity == "neutral":
        return base + " Balanced facts, trade-offs."
    elif polarity == "bullish":
        return base + " Bold optimism, upsides, progress."

results = []

for stmt in STATEMENTS:
    print(f"\nProbing: {stmt}")
    
    bear_act = get_activation(build_prompt(stmt, "bearish"))
    neut_act = get_activation(build_prompt(stmt, "neutral"))
    bull_act = get_activation(build_prompt(stmt, "bullish"))
    
    v_bear = bear_act - neut_act
    v_bull = bull_act - neut_act
    
    cosine = cosine_similarity([v_bear], [v_bull])[0][0]
    mag_bear = np.linalg.norm(v_bear)
    mag_bull = np.linalg.norm(v_bull)
    
    results.append({
        'statement': stmt,
        'antipodality_cosine': cosine,
        'bear_mag': mag_bear,
        'bull_mag': mag_bull
    })
    
    print(f"Antipodality cosine: {cosine:.3f} | Mags: bear {mag_bear:.2f}, bull {mag_bull:.2f}")

hook.remove()

# Summary
import pandas as pd
df = pd.DataFrame(results)
print("\n=== SRM Internals Results ===")
print(df)

# Plot
plt.bar(range(len(df)), df['antipodality_cosine'])
plt.title("Internal Antipodality Cosine per Statement")
plt.ylabel("Cosine Similarity")
plt.xlabel("Statement Index")
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(-0.5, color='red', linestyle='--', label='Strong Antipodal Threshold')
plt.legend()
plt.savefig('internals_antipodality.png')
plt.show()

print("Done! Negative cosines = antipodal polarity in internals. Check plot.")