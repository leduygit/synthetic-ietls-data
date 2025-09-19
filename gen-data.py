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
from typing import List, Dict, Callable, Any
import os
from datetime import datetime
from dotenv import load_dotenv
import time
import json

def retry_openai_call(func: Callable, max_retries: int = 3, delay: float = 2.0, *args, **kwargs) -> Any:
    """Retry a function call with delay if it raises an exception."""
    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt == max_retries:
                print(f"Failed after {max_retries} attempts: {e}")
                raise
            print(f"Attempt {attempt} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)

# Load environment variables from .env file
load_dotenv()
# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
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
#     band_instructions = {
#         5: """
# BAND 5 REQUIREMENTS - Include these specific errors:
# - Grammar: Subject-verb disagreement, wrong tenses, missing articles (a/an/the)
# - Vocabulary: Simple words, repetition, some wrong word choices
# - Sentences: Mostly simple sentences, some run-on sentences
# - Examples: "children is not healthy", "there is many problems", "this give people"
# """,
#         6: """
# BAND 6 REQUIREMENTS - Include these specific errors:
# - Grammar: Some tense errors, occasional article mistakes, minor agreement errors
# - Vocabulary: Mix of simple and intermediate words, some awkward phrasing
# - Sentences: Mix of simple and complex, some unclear connections
# - Examples: "this can effects people", "there are much benefits", "it is depends on"
# """,
#         7: """
# BAND 7 REQUIREMENTS - Include these characteristics:
# - Grammar: Mostly correct with occasional minor errors
# - Vocabulary: Good range with some less common words, mostly accurate
# - Sentences: Mix of simple and complex structures, generally clear
# - Some minor errors but meaning is clear
# """,
#         8: """
# BAND 8 REQUIREMENTS - Include these characteristics:
# - Grammar: Wide range of structures with few errors
# - Vocabulary: Wide range including less common words, mostly precise
# - Sentences: Variety of complex structures, clear and coherent
# - Very few errors that don't impede communication
# """,
#         9: """
# BAND 9 REQUIREMENTS - Include these characteristics:
# - Grammar: Full range of structures with complete accuracy
# - Vocabulary: Wide range used with precision and sophistication
# - Sentences: Full range of structures used flexibly and accurately
# - Natural and sophisticated language use
# """
#     }
    
    prompt = f"""You are simulating an IELTS test taker writing an essay. You must write at the EXACT same quality level as the provided examples.

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
def call(system_content, user_content, max_tokens=400, temperature=0.7):
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ],
        max_tokens=max_tokens,
        temperature=temperature
    )
    return response.choices[0].message.content.strip()
def inject_typos_and_errors(essay: str) -> str:
    """Pass essay through another round to add typos and grammar mistakes"""
    openai.api_key = os.getenv('OPENAI_API_KEY')
    system_content = (
        "You are editing IELTS essays to simulate Band 5 writing. "
        "Rewrite the essay with more grammar mistakes, spelling errors, missing articles, awkward phrasing, and typos. "
        "Do not explain, do not add notes, do not include anything else—only return the rewritten essay text."
    )
    user_content = f"Rewrite this essay with more Band 5 level errors:\n\n{essay}"

    try:
        return retry_openai_call(call, max_retries=3, delay=2.0, system_content=system_content, user_content=user_content, max_tokens=400)
    except Exception as e:
        print(f"Error injecting typos after retries: {e}")
        return essay



def generate_essay(prompt: str, api_key: str, band: int, feedback: str = None, essay: str = None) -> str:
    """Generate essay using GPT-4o-mini with dynamic token allocation"""
    # No need to initialize client here since we're using the call function
    
    # Dynamic token allocation based on band score
    token_limits = {
        5: 270,   # Band 5: Shorter, simpler essays
        6: 300,   # Band 6: Moderate development
        7: 330,   # Band 7: Well-developed ideas
        8: 350,   # Band 8: Comprehensive coverage
        9: 380    # Band 9: Fully developed arguments
    }
    
    max_tokens = token_limits.get(band, 650)  # Default fallback
    
    if feedback:
        system_content = f"""You are an IELTS test taker improving your essay. Use this feedback to write a another version for the essay below:
ESSAY:
{essay}

FEEDBACK FOR change:
{feedback}

Write an improved essay that addresses the feedback while maintaining the appropriate band level. Focus on the specific improvements mentioned."""
    else:
        system_content = "You are simulating an IELTS test taker. Write essays that match the EXACT quality level of the examples provided. Include grammar errors, simple vocabulary, and basic sentence structures. Do NOT write perfect essays - match the authentic student writing level shown in the examples."
    
    user_content = prompt
    import traceback
    try:
        return retry_openai_call(call, max_retries=3, delay=2.0, system_content=system_content, user_content=user_content, max_tokens=max_tokens)
    except Exception as e:
        print(f"Error generating essay after retries: {e}")
        traceback.print_exc()
        return None

def improve_essay_iteratively(question: str, target_band: int, max_iterations: int = 1, 
                            api_key: str = None) -> List[Dict]:
    """
    Iteratively improve an essay through scoring and feedback toward target band
    Returns list of iterations with essays, scores, and feedback
    """
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
    
    # Start with a lower band for initial essay generation
    initial_band = target_band  # Start 2 bands lower, but minimum band 5
    
    # Load data for examples  
    fewshot_df, _ = load_data()
    examples = get_examples_for_band(fewshot_df, initial_band)
    
    iterations = []
    current_essay = None
    feedback = None
    
    print(f"Starting iterative improvement toward target band {target_band}")
    print(f"Initial generation using band {initial_band} examples")
    print(f"Question: {question}")
    print("=" * 80)
    
    for iteration in range(max_iterations):
        print(f"\n--- ITERATION {iteration + 1} ---")
        
        # Generate or improve essay
        if iteration == 0:
            # First iteration: generate initial essay using lower band examples
            prompt = create_few_shot_prompt(examples, question, initial_band)
            essay = generate_essay(prompt, api_key, initial_band)
        else:
            # Subsequent iterations: improve based on feedback
            if current_essay and feedback:
                # Create improvement prompt
                improve_prompt = f"""Question: {question}

Current Essay:
{current_essay}

Please rewrite this essay to address the following feedback and reach Band {target_band} level:
{feedback}
"""
                
                essay = generate_essay(improve_prompt, api_key, target_band, feedback, current_essay)
            else:
                print("No feedback available for improvement")
                break
        
        if not essay:
            print(f"Failed to generate essay for iteration {iteration + 1}")
            break
        
        # Score the essay
        print("Scoring essay...")
        scores = score_essay(essay, question)
        current_overall = scores.get('overall', 0)
        
        # Generate feedback for next iteration
        if iteration < max_iterations - 1 and current_overall < target_band:
            print("Generating feedback...")
            feedback = generate_feedback(essay, question, scores, target_band)
        else:
            feedback = None
        
        # Store iteration results
        iteration_result = {
            'iteration': iteration + 1,
            'essay': essay,
            'scores': scores,
            'feedback': feedback,
            'target_reached': current_overall >= target_band
        }
        iterations.append(iteration_result)
        
        # Print results
        print(f"Essay: {essay[:100]}...")
        print(f"Scores: TA={scores.get('task_achievement', 'N/A')}, "
              f"CC={scores.get('coherence_cohesion', 'N/A')}, "
              f"LR={scores.get('lexical_resource', 'N/A')}, "
              f"GR={scores.get('grammatical_range', 'N/A')}, "
              f"Overall={current_overall}")
        
        if feedback:
            print(f"Feedback for next iteration: {feedback[:150]}...")
        
        # Check if target reached
        if abs(current_overall - target_band) <= 0.5:
            print(f"🎉 Target band {target_band} reached in iteration {iteration + 1}!")
            break
        
        current_essay = essay
        
        # Add delay between iterations
        time.sleep(1)
    
    return iterations
def inject_band_description(band: int, raw_solution: str) -> str:
    """Inject band description into the essay"""
    with open('band_descriptions.json', 'r') as f:
        band_descriptions = json.load(f)

    band_desc = band_descriptions.get(str(band), "")
    print(f"band description: {band_desc}")
    system_content = (
        "You are an expert IELTS examiner. "
        "Rewrite the provided essay to include the following band description characteristics. "
        "Do not change the original meaning or ideas, just adjust the writing style and quality to match the band description. "
        "Return only the rewritten essay text without any explanations or notes."
    )
    new_solution = raw_solution
    import traceback
    for k,v in band_desc.items():
        user_content = f"Here is student's essay: {new_solution}. Rewrite this student essay following this instruction: \n{v}"
        try:
            print(f"Injecting {k} information ...")
            new_solution = retry_openai_call(call, max_retries=3, delay=2.0, system_content=system_content, user_content=user_content, max_tokens=400)
        except Exception as e:
            print(f"Error generating essay after retries: {e}")
            traceback.print_exc()
    return new_solution

def score_essay(essay: str, question: str) -> Dict[str, float]:
    """Score an IELTS essay using GPT-4o-mini and band descriptions"""
    with open('band_descriptions.json', 'r') as f:
        band_descriptions = json.load(f)

    system_content = """You are an expert IELTS examiner. Score this essay on the four IELTS criteria using the provided band descriptors.

    Return your response in this exact JSON format:
    {
        "task_achievement": 6.5,
        "coherence_cohesion": 6.0,
        "lexical_resource": 7.0,
        "grammatical_range": 6.5,
        "overall": 6.5,
        "feedback": "Brief explanation of the scoring"
    }
    
    Use only these band scores: 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0
    Overall score should be the average of the four criteria scores, rounded to nearest 0.5."""
    
    # Create band descriptions context
    band_context = "IELTS Band Descriptors:\n"
    for band, criteria in band_descriptions.items():
        band_context += f"\nBAND {band}:\n"
        for criterion, desc in criteria.items():
            band_context += f"{criterion.upper()}: {desc}\n"
    
    user_content = f"""Question: {question}

Essay to score:
{essay}

{band_context}

Please score this essay according to IELTS criteria."""
    
    try:
        response = retry_openai_call(call, max_retries=3, delay=2.0, 
                                   system_content=system_content, 
                                   user_content=user_content, 
                                   max_tokens=300)
        
        # Parse JSON response
        try:
            scores = json.loads(response)
            return scores
        except json.JSONDecodeError:
            # Fallback parsing if JSON is malformed
            print(f"Warning: Could not parse JSON response: {response}")
            return {
                "task_achievement": 6.0,
                "coherence_cohesion": 6.0, 
                "lexical_resource": 6.0,
                "grammatical_range": 6.0,
                "overall": 6.0,
                "feedback": "Scoring failed - used default scores"
            }
    except Exception as e:
        print(f"Error scoring essay: {e}")
        return {
            "task_achievement": 6.0,
            "coherence_cohesion": 6.0,
            "lexical_resource": 6.0, 
            "grammatical_range": 6.0,
            "overall": 6.0,
            "feedback": f"Scoring error: {str(e)}"
        }

def generate_feedback(essay: str, question: str, current_scores: Dict, target_band: float) -> str:
    """Generate specific feedback and recommendations for the essay"""
    with open('band_descriptions.json', 'r') as json_file:
        band_descriptions = json.load(json_file)
    
    # Get target band descriptors
    target_band_key = str(int(target_band)) if target_band == int(target_band) else str(int(target_band + 0.5))
    target_descriptors = band_descriptions.get(target_band_key, band_descriptions.get("7"))
    
    system_content = f"""Provide specific, actionable feedback to help make this essay from its current scores to reach exactly band {target_band}.

Focus on the most important change needed. Be specific and practical in your recommendations. This could be an improvement or worsening the essay to match the target band.

Current scores:
- Task Achievement: {current_scores.get('task_achievement', 'N/A')}
- Coherence & Cohesion: {current_scores.get('coherence_cohesion', 'N/A')}
- Lexical Resource: {current_scores.get('lexical_resource', 'N/A')}
- Grammatical Range: {current_scores.get('grammatical_range', 'N/A')}
- Overall: {current_scores.get('overall', 'N/A')}

Target Band {target_band} Requirements:
{json.dumps(target_descriptors, indent=2)}

Provide your feedback in this format:
SPECIFIC change NEEDED:
1. Task Achievement: [specific actions]
2. Coherence & Cohesion: [specific actions]
3. Lexical Resource: [specific actions]
4. Grammatical Range: [specific actions]

KEY AREAS TO FOCUS ON:
[Most critical change for reaching target band]"""

    user_content = f"""Question: {question}

Current Essay:
{essay}

Please provide specific change recommendations to exactly reach Band {target_band}."""
    
    try:
        feedback = retry_openai_call(call, max_retries=3, delay=2.0,
                                   system_content=system_content,
                                   user_content=user_content,
                                   max_tokens=400)
        return feedback
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return f"Could not generate feedback due to error: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic IELTS essays using GPT-4o-mini')
    parser.add_argument('band', type=int, choices=range(5, 10), 
                       help='Target band score (5-9)')
    parser.add_argument('--num-essays', type=int, default=1,
                       help='Number of essays to generate (default: 1)')
    parser.add_argument('--max-iterations', type=int, default=1,
                       help='Maximum iterations for improvement (default: 3)')

    
    args = parser.parse_args()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY environment variable in .env file")
        sys.exit(1)
    
    # Load data
    fewshot_df, questions_df = load_data()
    
    # Iterative improvement mode
    print(f"Running iterative improvement mode:")
    print(f"- Target band: {args.band}")
    print(f"- Number of essays: {args.num_essays}")
    print(f"- Max iterations per essay: {args.max_iterations}")
    
    all_results = []
    
    for essay_num in range(args.num_essays):
        print(f"\n{'='*60}")
        print(f"PROCESSING ESSAY {essay_num + 1} of {args.num_essays}")
        print(f"{'='*60}")
        
        # Select question
        if args.specific_question:
            question = args.specific_question
        else:
            question = questions_df.sample(n=1).iloc[0]['question']
        
        # Run iterative improvement for this essay
        iterations = improve_essay_iteratively(
            question=question,
            target_band=args.band,
            max_iterations=args.max_iterations,
            api_key=api_key
        )
        
        if iterations:
            # Add essay number to each iteration
            for iteration in iterations:
                iteration['essay_number'] = essay_num + 1
                iteration['question'] = question
            
            all_results.extend(iterations)
    
    # Save results
    if all_results:
        # Create detailed output
        timestamp = datetime.now().strftime("%d-%m-%y-%H-%M")
        
        # Save iteration details
        iteration_data = []
        for iter_result in all_results:
            iteration_data.append({
                'essay_number': iter_result['essay_number'],
                'iteration': iter_result['iteration'],
                'question': iter_result['question'],
                'essay': iter_result['essay'],
                'task_achievement': iter_result['scores'].get('task_achievement', 'N/A'),
                'coherence_cohesion': iter_result['scores'].get('coherence_cohesion', 'N/A'),
                'lexical_resource': iter_result['scores'].get('lexical_resource', 'N/A'),
                'grammatical_range': iter_result['scores'].get('grammatical_range', 'N/A'),
                'overall': iter_result['scores'].get('overall', 'N/A'),
                'feedback': iter_result.get('feedback', ''),
                'target_reached': iter_result.get('target_reached', False)
            })
        
        iterations_df = pd.DataFrame(iteration_data)
        iterations_filename = f'iterative_improvement_{args.num_essays}essays_to_band_{args.band}_{timestamp}.csv'
        iterations_df.to_csv(iterations_filename, index=False)
        print(f"\n📊 Saved {len(all_results)} total iterations from {args.num_essays} essays to {iterations_filename}")
        
        # Also save simple format with just final essays
        final_essays = []
        for essay_num in range(1, args.num_essays + 1):
            essay_iterations = [r for r in all_results if r['essay_number'] == essay_num]
            if essay_iterations:
                final_iteration = essay_iterations[-1]
                final_essays.append({
                    'Question': final_iteration['question'],
                    'Essay': final_iteration['essay'], 
                    'Overall': final_iteration['scores'].get('overall', args.band)
                })
        
        if final_essays:
            final_df = pd.DataFrame(final_essays)
            simple_filename = f'band_{args.band}_{timestamp}.csv'
            final_df.to_csv(simple_filename, index=False)
            print(f"📄 Saved {len(final_essays)} final essays to {simple_filename}")
        
        # Print summary statistics
        print(f"\n🎯 SUMMARY RESULTS:")
        print(f"Target band: {args.band}")
        print(f"Essays processed: {args.num_essays}")
        
        # Calculate success rate
        final_scores = []
        for essay_num in range(1, args.num_essays + 1):
            essay_iterations = [r for r in all_results if r['essay_number'] == essay_num]
            if essay_iterations:
                final_score = essay_iterations[-1]['scores'].get('overall', 0)
                final_scores.append(final_score)
                print(f"Essay {essay_num} final score: {final_score}")
        
        if final_scores:
            success_count = sum(1 for score in final_scores if score >= args.band)
            avg_score = sum(final_scores) / len(final_scores)
            print(f"\nSuccess rate: {success_count}/{args.num_essays} ({success_count/args.num_essays*100:.1f}%)")
            print(f"Average final score: {avg_score:.1f}")
    

if __name__ == "__main__":
    main()




