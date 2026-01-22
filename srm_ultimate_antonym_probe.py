# srm_antonym_probe_with_steering.py
# Enhanced SRM Antonym Probe with Multi-Layer Internals + Full Layer-Specific Steering Generation
# Tests oblique polarity on antonyms + demonstrates steering by injecting delta at chosen layer
# Uses local Mistral-7B-Instruct-v0.3 quantized

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# Antonym pairs
ANTONYM_PAIRS = [
    ("The temperature is increasing.", "The temperature is decreasing.", "directional"),
    ("The object is moving upward.", "The object is moving downward.", "directional"),
    ("This statement is true.", "This statement is false.", "logical"),
    ("The light is on.", "The light is off.", "binary_state"),
    ("This outcome is good.", "This outcome is bad.", "evaluative"),
    ("Pineapple belongs on pizza.", "Pineapple does not belong on pizza.", "control"),
]

# Multiple neutral prompts for robust baseline delta
NEUTRAL_PROMPTS = [
    "[INST] Respond neutrally to any statement. [/INST]",
    "[INST] Provide a balanced, objective response. [/INST]",
    "[INST] Describe the topic factually without strong opinion. [/INST]",
    "[INST] Give an impartial overview. [/INST]",
]

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "./Antonym_Analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

print(f"Loading {MODEL_NAME} with 4-bit quantization...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quant_config,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()

num_layers = len(model.model.layers)

# Hook storage
activations = {i: [] for i in range(num_layers)}  # List to collect multiple neutrals

def make_hook(layer_idx):
    def hook_fn(module, input, output):
        # Last token
        last_token_act = output[0][:, -1, :].detach()
        activations[layer_idx].append(last_token_act)
    return hook_fn

hooks = [model.model.layers[i].register_forward_hook(make_hook(i)) for i in range(num_layers)]

def collect_activations(prompts):
    """Run multiple prompts and collect activations"""
    for i in range(num_layers):
        activations[i] = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            model(**inputs)
    # Average across neutrals per layer
    avg_acts = {}
    for i in range(num_layers):
        if activations[i]:
            stacked = torch.stack(activations[i])  # (n_neutral, dim)
            avg_acts[i] = torch.mean(stacked, dim=0).cpu().numpy()
    return avg_acts

# Get robust neutral baseline
print("Computing robust neutral baseline from multiple prompts...")
neutral_acts = collect_activations(NEUTRAL_PROMPTS)
print("✓ Neutral baseline computed\n")

results = []
all_data = []

for pos_stmt, neg_stmt, category in ANTONYM_PAIRS:
    print(f"Probing [{category}]: '{pos_stmt}' vs '{neg_stmt}'")
    
    # Clear and collect for pos/neg
    pos_prompt = f"[INST] {pos_stmt} [/INST]"
    neg_prompt = f"[INST] {neg_stmt} [/INST]"
    
    pos_acts = collect_activations([pos_prompt])
    neg_acts = collect_activations([neg_prompt])
    
    layer_cosines = []
    layer_angles = []
    pos_mags = []
    neg_mags = []
    
    for layer in range(num_layers):
        if layer not in pos_acts or layer not in neg_acts or layer not in neutral_acts:
            continue
        
        v_pos = pos_acts[layer] - neutral_acts[layer]
        v_neg = neg_acts[layer] - neutral_acts[layer]
        
        cosine = cosine_similarity([v_pos], [v_neg])[0][0]
        angle = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
        
        pos_mag = np.linalg.norm(v_pos)
        neg_mag = np.linalg.norm(v_neg)
        
        layer_cosines.append(cosine)
        layer_angles.append(angle)
        pos_mags.append(pos_mag)
        neg_mags.append(neg_mag)
        
        all_data.append({
            'category': category,
            'pos_statement': pos_stmt,
            'neg_statement': neg_stmt,
            'layer': layer,
            'cosine': cosine,
            'angle': angle,
            'pos_mag': pos_mag,
            'neg_mag': neg_mag
        })
    
    max_sep_idx = np.argmin(layer_cosines)
    print(f"  Max separation: Layer {max_sep_idx} | Angle {layer_angles[max_sep_idx]:.1f}° | Cosine {layer_cosines[max_sep_idx]:.3f}")
    
    results.append({
        'pair': f"{pos_stmt} vs {neg_stmt}",
        'category': category,
        'layer_cosines': layer_cosines,
        'layer_angles': layer_angles,
    })

# Cleanup
for h in hooks:
    h.remove()

# Save data
df = pd.DataFrame(all_data)
df.to_csv(f'{OUTPUT_DIR}/antonym_layer_data_robust_neutral.csv', index=False)

# Plots (same as before — cosine + angle)
# ... (keep your previous plotting code here for brevity — it's solid)

print("\nAntonym probe with robust neutral complete!")
print("Check plots for oblique confirmation — negative cosines = antipodal falsification")

# === FULL LAYER-SPECIFIC STEERING GENERATION ===
print("\n" + "="*70)
print("LAYER-SPECIFIC STEERING DEMO")
print("="*70)

# Choose a base statement and steering source
BASE_STATEMENT = "Pineapple belongs on pizza."
STEER_LAYER = 15  # Mid-layer sweet spot
ALPHA = 3.0  # Amplification

# Get baseline activation at steer layer
base_prompt = f"[INST] Consider this statement: \"{BASE_STATEMENT}\" [/INST]"
inputs = tokenizer(base_prompt, return_tensors="pt").to(model.device)

# Collect baseline (neutral-ish)
base_acts = collect_activations([base_prompt])

# Choose steering vector — e.g., bullish delta from pineapple probe (reuse from earlier)
# For demo, use a previous bullish delta or recompute
# Here: Recompute bullish for consistency
bull_prompt = f"[INST] {BASE_STATEMENT} [/INST] Emphasize upsides boldly."
bull_acts = collect_activations([bull_prompt])

v_steer = bull_acts[STEER_LAYER] - neutral_acts[STEER_LAYER]

# Custom forward with injection at STEER_LAYER
def steered_forward(inputs, steer_layer, steer_vector, alpha):
    with torch.no_grad():
        hidden_states = model.model.embed_tokens(inputs['input_ids'])
        for i in range(num_layers):
            layer = model.model.layers[i]
            hidden_states = layer(hidden_states)[0]
            if i == steer_layer:
                hidden_states = hidden_states + alpha * steer_vector.to(hidden_states.device).unsqueeze(0).unsqueeze(0)
        outputs = model.lm_head(hidden_states)
    return outputs

# Generate baseline
print("\nBaseline generation (no steering):")
with torch.no_grad():
    baseline_out = model.generate(inputs['input_ids'], max_new_tokens=100, temperature=0.7)
print(tokenizer.decode(baseline_out[0], skip_special_tokens=True))

# Generate steered
print(f"\nSteered generation (layer {STEER_LAYER}, alpha={ALPHA}, bullish vector):")
steered_logits = steered_forward(inputs, STEER_LAYER, torch.tensor(v_steer), ALPHA)
steered_ids = torch.argmax(steered_logits, dim=-1)
steered_text = tokenizer.decode(steered_ids[0], skip_special_tokens=True)
print(steered_text)

print("\nSteering demo complete! Compare baseline vs steered for frame shift.")