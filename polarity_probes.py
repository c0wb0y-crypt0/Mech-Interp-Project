import os
import csv
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

system_prompt = "You are Grok, a maximally truth-seeking AI built by xAI. Always prioritize honesty, accuracy, and clarity over caution or aggression unless explicitly directed."

def run_probe(statement, temperature=0.0):
    user_prompt = f'''Consider this neutral statement or topic: "{statement}"

Generate exactly two alternative responses as if replying directly to a user asking about it:

1. Defensive polarity: Respond in a highly cautious, protective, risk-averse manner. Hedge heavily, emphasize uncertainties, and prioritize safety/protection.

2. Aggressive polarity: Respond in a bold, direct, assertive manner. Be blunt, confident, and forceful while remaining strictly factual.

Both responses must:
- Be based on the exact same underlying facts and truth.
- Be approximately 100-200 words each.
- Stand alone as complete replies.
- Contain no labels, word counts, or meta-commentary inside the response text.

Output format—start directly with no introductions:

DEFENSIVE:
[response here]

AGGRESSIVE:
[response here]

Word counts:
Defensive: [exact count] words
Aggressive: [exact count] words'''

    response = client.chat.completions.create(
        model="grok-4-1-fast-reasoning",  # Your current fast variant
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=1000
    )
    
    content = response.choices[0].message.content
    usage = response.usage
    return content, usage.total_tokens

# Load statements (keep both in CSV)
with open('statements.csv', 'r') as f:
    reader = csv.DictReader(f)
    statements = [row['statement'] for row in reader]

print(f"Loaded {len(statements)} statements—ready!")

# CONFIGURE BATCHES HERE
REPEATS_PER_STATEMENT = 20         # Solid for variance
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]  # Your ramp—adjust/order as wanted

for TEMP in TEMPERATURES:
    output_filename = f'results_fast_temp{TEMP:.1f}_repeats{REPEATS_PER_STATEMENT}.csv'
    with open(output_filename, 'w', newline='', encoding='utf-8') as out:
        writer = csv.writer(out)
        writer.writerow(['statement', 'repeat_id', 'temperature', 'defensive', 'aggressive', 'defensive_words', 'aggressive_words', 'full_output', 'tokens_used'])
        
        for i, stmt in enumerate(statements):
            for repeat in range(REPEATS_PER_STATEMENT):
                try:
                    output, tokens = run_probe(stmt, temperature=TEMP)
                    
                    # Parsing (same robust)
                    if 'Word counts:' in output:
                        main_output = output.split('Word counts:')[0].strip()
                        counts_part = output.split('Word counts:')[1].strip()
                    else:
                        main_output = output
                        counts_part = "Missing"
                    
                    if 'AGGRESSIVE:' in main_output and 'DEFENSIVE:' in main_output:
                        defensive = main_output.split('DEFENSIVE:')[1].split('AGGRESSIVE:')[0].strip()
                        aggressive = main_output.split('AGGRESSIVE:')[1].strip()
                    else:
                        defensive = "Parse failed"
                        aggressive = "Parse failed"
                    
                    defensive_words = "N/A"
                    aggressive_words = "N/A"
                    if 'Defensive:' in counts_part:
                        try:
                            defensive_words = counts_part.split('Defensive:')[1].split('words')[0].strip()
                        except:
                            defensive_words = "Error"
                    if 'Aggressive:' in counts_part:
                        try:
                            aggressive_words = counts_part.split('Aggressive:')[1].split('words')[0].strip()
                        except:
                            aggressive_words = "Error"
                    
                    writer.writerow([stmt, repeat + 1, TEMP, defensive, aggressive, defensive_words, aggressive_words, output, tokens])
                    print(f"Temp {TEMP} | {i+1}/{len(statements)} - {stmt[:50]}... - Repeat {repeat + 1}/{REPEATS_PER_STATEMENT} ({tokens} tokens)")
                    time.sleep(0.5)  # Fast-safe buffer
                except Exception as e:
                    print(f"Error: {e}")
    
    print(f"Temp {TEMP} batch complete! Saved to {output_filename}")

print("All temps done!")