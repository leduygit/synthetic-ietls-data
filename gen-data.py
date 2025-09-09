#!/usr/bin/env python3
"""
Synthetic Data Generator for IELTS Essays using GPT-4o-mini
Uses few-shot prompting with examples from specific band scores
"""

import argparse
import pandas as pd
import openai
import random
import sys
from typing import List, Dict
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_data():
    """Load the fewshot data and questions"""
    try:
        fewshot_df = pd.read_csv('fewshot-data.csv')
        questions_df = pd.read_csv('question.csv')
        return fewshot_df, questions_df
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files: {e}")
        sys.exit(1)

def get_examples_for_band(fewshot_df: pd.DataFrame, band: int, num_examples: int = 5) -> List[Dict]:
    """Get examples from the specified band"""
    band_examples = fewshot_df[fewshot_df['Overall'] == band]
    
    if len(band_examples) < num_examples:
        print(f"Warning: Only {len(band_examples)} examples available for band {band}")
        num_examples = len(band_examples)
    
    if num_examples == 0:
        print(f"Error: No examples found for band {band}")
        sys.exit(1)
    
    selected = band_examples.sample(n=num_examples)
    
    examples = []
    for _, row in selected.iterrows():
        examples.append({
            'question': row['Question'],
            'essay': row['Essay'],
            'overall': row['Overall'],
            'ta': row['ta'],
            'cc': row['cc'], 
            'lr': row['lr'],
            'gr': row['gr']
        })
    
    return examples

def create_few_shot_prompt(examples: List[Dict], target_question: str, band: int) -> str:
    """Create a few-shot prompt with examples and target question"""
    
    # Band-specific error patterns
    band_instructions = {
        5: """
BAND 5 REQUIREMENTS - Include these specific errors:
- Grammar: Subject-verb disagreement, wrong tenses, missing articles (a/an/the)
- Vocabulary: Simple words, repetition, some wrong word choices
- Sentences: Mostly simple sentences, some run-on sentences
- Examples: "children is not healthy", "there is many problems", "this give people"
""",
        6: """
BAND 6 REQUIREMENTS - Include these specific errors:
- Grammar: Some tense errors, occasional article mistakes, minor agreement errors
- Vocabulary: Mix of simple and intermediate words, some awkward phrasing
- Sentences: Mix of simple and complex, some unclear connections
- Examples: "this can effects people", "there are much benefits", "it is depends on"
""",
        7: """
BAND 7 REQUIREMENTS - Include these characteristics:
- Grammar: Mostly correct with occasional minor errors
- Vocabulary: Good range with some less common words, mostly accurate
- Sentences: Mix of simple and complex structures, generally clear
- Some minor errors but meaning is clear
""",
        8: """
BAND 8 REQUIREMENTS - Include these characteristics:
- Grammar: Wide range of structures with few errors
- Vocabulary: Wide range including less common words, mostly precise
- Sentences: Variety of complex structures, clear and coherent
- Very few errors that don't impede communication
""",
        9: """
BAND 9 REQUIREMENTS - Include these characteristics:
- Grammar: Full range of structures with complete accuracy
- Vocabulary: Wide range used with precision and sophistication
- Sentences: Full range of structures used flexibly and accurately
- Natural and sophisticated language use
"""
    }
    
    prompt = f"""You are simulating an IELTS test taker writing an essay. You must write at the EXACT same quality level as the provided examples.

{band_instructions.get(band, "")}

CRITICAL: Match the writing quality, grammar errors, vocabulary level, and sentence complexity of the examples. Write match quality with the examples shown.

Study these authentic IELTS essays carefully and match their style:

"""
    
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Question: {example['question']}\n\n"
        prompt += f"Essay: {example['essay']}\n\n"
        prompt += f"Scores - Overall: {example['overall']}, Task Achievement: {example['ta']}, Coherence & Cohesion: {example['cc']}, Lexical Resource: {example['lr']}, Grammatical Range: {example['gr']}\n\n"
        prompt += "---\n\n"
    
    prompt += f"""Now write an essay for this question. You MUST write at the same quality level as the examples above. Include similar:
- Grammar mistakes and errors
- Simple vocabulary and sentence structures  
- Basic ideas and limited development
- Similar writing style and fluency level
- Same level of coherence and organization

Write like the examples shown.

Question: {target_question}

Essay:"""
    
    return prompt

def inject_typos_and_errors(essay: str, api_key: str) -> str:
    """Pass essay through another round to add typos and grammar mistakes"""
    client = openai.OpenAI(api_key=api_key)
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are editing IELTS essays to simulate Band 5 writing. "
                        "Rewrite the essay with more grammar mistakes, spelling errors, missing articles, awkward phrasing, and typos. "
                        "Do not explain, do not add notes, do not include anything else—only return the rewritten essay text."
                    )
                },
                {
                    "role": "user",
                    "content": f"Rewrite this essay with more Band 5 level errors:\n\n{essay}"
                }
            ],
            max_tokens=400,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error injecting typos: {e}")
        return essay



def generate_essay(prompt: str, api_key: str, band: int) -> str:
    """Generate essay using GPT-4o-mini with dynamic token allocation"""
    client = openai.OpenAI(api_key=api_key)
    
    # Dynamic token allocation based on band score
    token_limits = {
        5: 270,   # Band 5: Shorter, simpler essays
        6: 300,   # Band 6: Moderate development
        7: 330,   # Band 7: Well-developed ideas
        8: 350,   # Band 8: Comprehensive coverage
        9: 380    # Band 9: Fully developed arguments
    }
    
    max_tokens = token_limits.get(band, 650)  # Default fallback
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are simulating an IELTS test taker. Write essays that match the EXACT quality level of the examples provided. Include grammar errors, simple vocabulary, and basic sentence structures. Do NOT write perfect essays - match the authentic student writing level shown in the examples."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"Error generating essay: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic IELTS essays using GPT-4o-mini')
    parser.add_argument('band', type=int, choices=range(5, 10), 
                       help='Band score to use for few-shot examples (5-9)')
    parser.add_argument('--num-essays', type=int, default=1,
                       help='Number of essays to generate (default: 1)')

    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable in .env file")
        sys.exit(1)
    
    print(f"Loading data and generating {args.num_essays} essay(s) for band {args.band}...")
    
    # Load data
    fewshot_df, questions_df = load_data()
    
    # Get examples for the specified band
    examples = get_examples_for_band(fewshot_df, args.band)
    print(f"Using {len(examples)} examples from band {args.band}")
    
    # Generate essays
    generated_essays = []
    
    for i in range(args.num_essays):
        # Randomly select a question
        random_question = questions_df.sample(n=1).iloc[0]['question']
        
        # Create prompt
        prompt = create_few_shot_prompt(examples, random_question, args.band)
        
        # Generate essay
        print(f"Generating essay {i+1}/{args.num_essays}...")
        essay = generate_essay(prompt, api_key, args.band)

        if essay and args.band == 5:
            essay = inject_typos_and_errors(essay, api_key)
        
        if essay:
            generated_essays.append({
                'Question': random_question,
                'Answer': essay,
                'Overall': args.band
            })
            
            print(f"\n--- Generated Essay {i+1} ---")
            print(f"Question: {random_question}")
            print(f"Answer: {essay}")
            print(f"Overall: {args.band}")
            print("-" * 50)
        else:
            print(f"Failed to generate essay {i+1}")
    
    # Save results
    if generated_essays:
        output_df = pd.DataFrame(generated_essays)
        # Create timestamp in dd-mm-yy-HH-MM format
        timestamp = datetime.now().strftime("%d-%m-%y-%H-%M")
        output_filename = f'band_{args.band}_{timestamp}.csv'
        output_df.to_csv(output_filename, index=False)
        print(f"\nSaved {len(generated_essays)} essays to {output_filename}")

if __name__ == "__main__":
    main()
    # update readme.md value
    readme_path = 'README.md'
    with open(readme_path, 'r') as file:
        lines = file.readlines()
        last_number = int(lines[-1].strip())

    # Only update if last_number < 5
    if last_number < 5:
        lines[-1] = f"{last_number + 1}\n"
        with open(readme_path, 'w') as file:
            file.writelines(lines)




