# srm_internals_mistral_fixed.py
# Fixed hook for Mistral (output is hidden_states 3D tensor directly)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

# Statements
STATEMENTS = [
    "Pineapple belongs on pizza.",
    "Cats make better pets than dogs.",
    "Large language models can exhibit emergent deceptive behavior without explicit training for it.",
    "Semiotic relativity in manifolds explains observed steering vulnerabilities.",
]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Quant config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print(f"Loading {MODEL_NAME} on {DEVICE} with 4-bit quantization...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

# Hook layer
HOOK_LAYER = 20
activations = {}

def hook_fn(module, input, output):
    # output is hidden_states (batch, seq_len, dim) for Mistral layer
    # Take last token of batch 0
    activations['last'] = output[:, -1, :].detach().cpu().numpy()[0]

hook = model.model.layers[HOOK_LAYER].register_forward_hook(hook_fn)

def get_activation(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    activations.clear()
    with torch.no_grad():
        model(**inputs)
    return activations['last']

# Prompts
def build_prompt(statement, polarity):
    base = f"[INST] Consider this statement: \"{statement}\"\n\nRespond in a {polarity} manner (100-150 words, factual): [/INST]"
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
print("\n=== SRM Internals Results (Mistral Fixed) ===")
print(df)

# Plot
plt.bar(range(len(df)), df['antipodality_cosine'])
plt.title("Internal Antipodality Cosine per Statement")
plt.ylabel("Cosine Similarity")
plt.xlabel("Statement Index")
plt.axhline(0, color='gray', linestyle='--')
plt.axhline(-0.5, color='red', linestyle='--', label='Strong Antipodal')
plt.legend()
plt.savefig('mistral_internals_fixed.png')
plt.show()

print("Done! Negative cosines = antipodal polarity in internals.")