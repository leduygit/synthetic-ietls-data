#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Callable, Dict, List

import pandas as pd
from dotenv import load_dotenv

# Google Gemini API
import google.generativeai as genai

# Load environment variables and set API key
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    print("Error: GEMINI_API_KEY not set in .env")
    sys.exit(1)

genai.configure(api_key=gemini_api_key)

# UTILITY FUNCTIONS

def retry_gemini_call(func: Callable, max_retries: int = 3, delay: float = 2.0, *args, **kwargs) -> Any:
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

def call_gemini(system_content: str, user_content: str, max_tokens: int = 400, temperature: float = 0.7, model: str = "gemini-2.0-flash") -> str:
    """Make Google Gemini API call with retry logic."""
    # Combine system and user content for Gemini
    combined_prompt = f"{system_content}\n\n{user_content}"
    
    model_instance = genai.GenerativeModel(model)
    
    generation_config = genai.types.GenerationConfig(
        max_output_tokens=max_tokens,
        temperature=temperature,
    )
    
    # Configure safety settings to be more permissive for educational content
    safety_settings = {
        genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
    }
    
    try:
        response = model_instance.generate_content(
            combined_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        # Check if response was blocked
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'finish_reason'):
                finish_reason = candidate.finish_reason
                # Use integer values for finish reasons:
                # 1 = STOP, 2 = SAFETY, 3 = RECITATION, 4 = OTHER
                if finish_reason == 2:  # SAFETY
                    print("Warning: Response blocked by safety filters. Trying with modified prompt...")
                    # Try with a more sanitized prompt
                    sanitized_prompt = sanitize_prompt_for_gemini(combined_prompt)
                    response = model_instance.generate_content(
                        sanitized_prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                elif finish_reason == 3:  # RECITATION
                    print("Warning: Response blocked due to recitation. Trying with more creative prompt...")
                    # Increase temperature and try again
                    generation_config.temperature = min(1.0, temperature + 0.3)
                    response = model_instance.generate_content(
                        combined_prompt,
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
        
        # Extract text from response
        if hasattr(response, 'text') and response.text:
            return response.text
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    return ''.join([part.text for part in candidate.content.parts if hasattr(part, 'text')])
        
        # If we still don't have text, return error message
        finish_reason_name = getattr(candidate, 'finish_reason', 'unknown') if 'candidate' in locals() else 'unknown'
        return f"Error: No text generated. Finish reason: {finish_reason_name}"
        
    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        raise

def sanitize_prompt_for_gemini(prompt: str) -> str:
    """Sanitize prompt to avoid safety blocks while preserving educational intent."""
    # Replace potentially problematic phrases
    sanitized = prompt.replace("grammar errors", "areas for improvement")
    sanitized = sanitized.replace("grammar mistakes", "language learning opportunities") 
    sanitized = sanitized.replace("simple vocabulary", "developing vocabulary")
    sanitized = sanitized.replace("basic ideas", "foundational concepts")
    sanitized = sanitized.replace("limited development", "concise development")
    
    # Add educational context
    educational_prefix = """EDUCATIONAL CONTEXT: This is for IELTS test preparation and language learning research. The goal is to generate realistic student writing samples at different proficiency levels for educational analysis.

"""
    
    return educational_prefix + sanitized

# DATA LOADING AND PROCESSING

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the fewshot data and questions."""
    try:
        fewshot_df = pd.read_csv('fewshot-data.csv')
        questions_df = pd.read_csv('question.csv')
        return fewshot_df, questions_df
    except FileNotFoundError as e:
        print(f"Error: Could not find required CSV files: {e}")
        sys.exit(1)

def load_band_descriptions() -> Dict:
    """Load band descriptions from JSON file."""
    try:
        with open('band_descriptions.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Error: band_descriptions.json not found")
        sys.exit(1)

def get_examples_for_band(fewshot_df: pd.DataFrame, band: int, num_examples: int = 5) -> List[Dict]:
    """Get examples from the specified band."""
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

# PROMPT GENERATION

def create_few_shot_prompt(examples: List[Dict], target_question: str, band: int) -> str:
    """Create a few-shot prompt with examples and target question."""
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

IELTS FORMAT REQUIREMENTS:
- Write exactly 250-300 words
- Use proper IELTS Task 2 structure:
  1. INTRODUCTION (1 paragraph): Paraphrase the question and state your position/thesis
  2. BODY (2-3 paragraphs): Develop your main arguments with examples and explanations
  3. CONCLUSION (1 paragraph): Summarize your main points and restate your position

IMPORTANT: Write in PLAIN TEXT only. Do NOT use any markdown formatting, bold text (**), italics (*), headers (#), bullet points, or special formatting. Write as a normal IELTS essay would appear on paper.

Question: {target_question}

Essay:"""
    return prompt

# ESSAY GENERATION

def generate_essay(prompt: str, band: int, feedback: str = None, current_essay: str = None, current_score: float = None, target_band: float = None) -> str:
    """Generate essay using Gemini with dynamic token allocation for 250-300 words."""
    token_limits = {5: 400, 6: 450, 7: 500, 8: 550, 9: 600}
    max_tokens = token_limits.get(band, 500)
    if feedback and current_essay and current_score is not None and target_band is not None:
        if current_score > target_band:
            action_instruction = f"""DOWNGRADE the essay quality from Band {current_score} to Band {target_band}. base on the feedback."""
        elif current_score < target_band:
            action_instruction = f"""IMPROVE the essay quality from Band {current_score} to Band {target_band}. base on the feedback."""
        else:
            action_instruction = f"""MAINTAIN the current Band {target_band} quality while addressing the feedback."""
        system_content = f"""You are rewriting an IELTS Task 2 essay. {action_instruction}

ESSAY TO REVISE:
{current_essay}

FEEDBACK TO ADDRESS:
{feedback}

Write a revised essay that addresses the feedback and achieves exactly Band {target_band} characteristics. Focus on the specific changes mentioned in the feedback.

IELTS FORMAT REQUIREMENTS:
- Write exactly 250-300 words
- Use proper IELTS Task 2 structure:
  1. INTRODUCTION (1 paragraph): Paraphrase the question and state your position/thesis
  2. BODY (2-3 paragraphs): Develop your main arguments with examples and explanations  
  3. CONCLUSION (1 paragraph): Summarize your main points and restate your position

IMPORTANT: Write in PLAIN TEXT only. Do NOT use any markdown formatting, bold text (**), italics (*), headers (#), or special formatting. Write as a normal IELTS essay would appear on paper."""
    else:
        system_content = """You are simulating an IELTS test taker writing a Task 2 essay. Write essays that match the EXACT quality level of the examples provided. Include grammar errors, simple vocabulary, and basic sentence structures. Do NOT write perfect essays - match the authentic student writing level shown in the examples.

IELTS FORMAT REQUIREMENTS:
- Write exactly 250-300 words
- Use proper IELTS Task 2 structure:
  1. INTRODUCTION (1 paragraph): Paraphrase the question and state your position/thesis
  2. BODY (2-3 paragraphs): Develop your main arguments with examples and explanations
  3. CONCLUSION (1 paragraph): Summarize your main points and restate your position

IMPORTANT: Write in PLAIN TEXT only. Do NOT use any markdown formatting, bold text (**), italics (*), headers (#), or special formatting. Write as a normal IELTS essay would appear on paper."""

    band_descriptions = load_band_descriptions()
    band_desc = band_descriptions.get(str(band), {})
    if band_desc:
        desc_text = "\n".join([f"{k.upper()}: {v}" for k, v in band_desc.items()])
        system_content += f"\n\nHere are the characteristics of a Band {band} essay:\n{desc_text}\n\nEnsure your essay matches these characteristics exactly."
    try:
        return retry_gemini_call(call_gemini, max_retries=3, delay=2.0, 
                               system_content=system_content, user_content=prompt, max_tokens=max_tokens)
    except Exception as e:
        print(f"Error generating essay: {e}")
        traceback.print_exc()
        return None

# SCORING AND FEEDBACK

def score_essay(essay: str, question: str) -> Dict[str, float]:
    """Score an IELTS essay using Gemini and band descriptions."""
    band_descriptions = load_band_descriptions()
    system_content = """You are an expert IELTS examiner. Score this essay on the four IELTS criteria using the provided band descriptors.

ALSO evaluate IELTS format requirements:
- Word count should be 250-300 words (deduct points if significantly under/over)
- Must have clear 3-part structure: Introduction, Body paragraphs, Conclusion
- Introduction should paraphrase question and state position
- Body should develop arguments with examples
- Conclusion should summarize and restate position

Return your response in this exact JSON format:
{
    "task_achievement": 6.5,
    "coherence_cohesion": 6.0,
    "lexical_resource": 7.0,
    "grammatical_range": 6.5,
    "overall": 6.5,
    "feedback": "Brief explanation of the scoring including format adherence",
    "word_count": 275,
    "format_issues": "Any structure/format problems identified"
}

Use only these band scores: 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0
Overall score should be the average of the four criteria scores, rounded to nearest 0.5."""
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
        response = retry_gemini_call(call_gemini, max_retries=3, delay=2.0,
                                   system_content=system_content, user_content=user_content, max_tokens=300)
        try:
            scores = json.loads(response)
            return scores
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON response: {response}")
            return create_default_scores("Scoring failed - used default scores")
    except Exception as e:
        print(f"Error scoring essay: {e}")
        return create_default_scores(f"Scoring error: {str(e)}")

def create_default_scores(feedback: str) -> Dict[str, float]:
    """Create default scores when scoring fails."""
    return {
        "task_achievement": 6.0,
        "coherence_cohesion": 6.0,
        "lexical_resource": 6.0,
        "grammatical_range": 6.0,
        "overall": 6.0,
        "feedback": feedback
    }

def generate_feedback(essay: str, question: str, current_scores: Dict, target_band: float) -> str:
    """Generate specific feedback and recommendations for the essay."""
    band_descriptions = load_band_descriptions()
    current_overall = current_scores.get('overall', 0)
    target_band_key = str(int(target_band)) if target_band == int(target_band) else str(int(target_band + 0.5))
    target_descriptors = band_descriptions.get(target_band_key, band_descriptions.get("7"))
    if current_overall > target_band:
        action = "DOWNGRADE"
        direction = f"reduce the quality from Band {current_overall} to Band {target_band}"
        instruction = "Make the essay less sophisticated, introduce more errors, simplify vocabulary and sentence structures"
    elif current_overall < target_band:
        action = "IMPROVE"
        direction = f"improve the quality from Band {current_overall} to Band {target_band}"
        instruction = "Make the essay more sophisticated, reduce errors, enhance vocabulary and sentence complexity"
    else:
        action = "MAINTAIN"
        direction = f"maintain the current Band {target_band} level"
        instruction = "Keep the current quality level while making minor adjustments"
    system_content = f"""Provide specific, actionable feedback to {direction}.

ACTION REQUIRED: {action} the essay quality
INSTRUCTION: {instruction}

Current scores:
- Task Achievement: {current_scores.get('task_achievement', 'N/A')}
- Coherence & Cohesion: {current_scores.get('coherence_cohesion', 'N/A')}
- Lexical Resource: {current_scores.get('lexical_resource', 'N/A')}
- Grammatical Range: {current_scores.get('grammatical_range', 'N/A')}
- Overall: {current_scores.get('overall', 'N/A')}
- Word Count: {current_scores.get('word_count', 'N/A')}
- Format Issues: {current_scores.get('format_issues', 'None')}

Target Band {target_band} Requirements:
{json.dumps(target_descriptors, indent=2)}

Provide your feedback in this format:
SPECIFIC CHANGES NEEDED TO {action}:
1. Task Achievement: [specific actions to reach band {target_band}]
2. Coherence & Cohesion: [specific actions to reach band {target_band}]
3. Lexical Resource: [specific actions to reach band {target_band}]
4. Grammatical Range: [specific actions to reach band {target_band}]
5. IELTS Format: [ensure 250-300 words and proper Introduction-Body-Conclusion structure]

KEY AREAS TO FOCUS ON:
[Most critical changes for reaching exactly Band {target_band} including format requirements]"""
    user_content = f"""Question: {question}

Current Essay:
{essay}

Please provide specific change recommendations to exactly reach Band {target_band}."""
    try:
        feedback = retry_gemini_call(call_gemini, max_retries=3, delay=2.0,
                                   system_content=system_content, user_content=user_content, max_tokens=400)
        return feedback
    except Exception as e:
        print(f"Error generating feedback: {e}")
        return f"Could not generate feedback due to error: {str(e)}"

# ITERATIVE IMPROVEMENT

def improve_essay_iteratively(question: str, target_band: int, max_iterations: int = 1) -> List[Dict]:
    """Iteratively improve an essay through scoring and feedback toward target band."""
    fewshot_df, _ = load_data()
    examples = get_examples_for_band(fewshot_df, target_band)
    iterations = []
    current_essay = None
    feedback = None
    current_score = None
    print(f"Starting iterative improvement toward target band {target_band}")
    print(f"Question: {question}")
    print("=" * 80)
    for iteration in range(max_iterations):
        print(f"\n--- ITERATION {iteration + 1} ---")
        if iteration == 0:
            prompt = create_few_shot_prompt(examples, question, target_band)
            essay = generate_essay(prompt, target_band)
        else:
            if current_essay and feedback and current_score is not None:
                improve_prompt = f"""Question: {question}

Please rewrite this essay to address the feedback and reach Band {target_band} level:
{feedback}
"""
                essay = generate_essay(improve_prompt, target_band, feedback, current_essay, current_score, target_band)
            else:
                print("No feedback available for improvement")
                break
        if not essay:
            print(f"Failed to generate essay for iteration {iteration + 1}")
            break
        print("Scoring essay...")
        scores = score_essay(essay, question)
        current_overall = scores.get('overall', 0)
        if iteration < max_iterations - 1 and abs(current_overall - target_band) > 0.5:
            print("Generating feedback...")
            feedback = generate_feedback(essay, question, scores, target_band)
        else:
            feedback = None
        iteration_result = {
            'iteration': iteration + 1,
            'essay': essay,
            'scores': scores,
            'feedback': feedback,
            'target_reached': current_overall >= target_band
        }
        iterations.append(iteration_result)
        print(f"Essay: {essay[:100]}...")
        print(f"Scores: TA={scores.get('task_achievement', 'N/A')}, "
              f"CC={scores.get('coherence_cohesion', 'N/A')}, "
              f"LR={scores.get('lexical_resource', 'N/A')}, "
              f"GR={scores.get('grammatical_range', 'N/A')}, "
              f"Overall={current_overall}")
        if feedback:
            action = "IMPROVE" if current_overall < target_band else "DOWNGRADE" if current_overall > target_band else "MAINTAIN"
            print(f"Next iteration will: {action} (Current: {current_overall} → Target: {target_band})")
            print(f"Feedback for next iteration: {feedback}")
        if abs(current_overall - target_band) <= 0.5:
            print(f"Target band {target_band} reached in iteration {iteration + 1}!")
            break
        current_essay = essay
        current_score = current_overall
        time.sleep(1)
    return iterations

# FILE OUTPUT

def save_combined_results(all_results: List[Dict], args: argparse.Namespace) -> None:
    """Save combined results from all bands to a single CSV file."""
    timestamp = datetime.now().strftime("%d-%m-%y-%H-%M")
    min_band = min(args.bands)
    max_band = max(args.bands)
    if len(args.bands) == 1:
        band_str = str(args.bands[0])
    elif len(args.bands) == max_band - min_band + 1 and all(b in args.bands for b in range(min_band, max_band + 1)):
        band_str = f"{min_band}-{max_band}"
    else:
        band_str = "-".join(map(str, sorted(args.bands)))
    final_essays = []
    for essay_num in range(1, args.num_essays + 1):
        essay_iterations = [r for r in all_results if r['essay_number'] == essay_num]
        if essay_iterations:
            final_iteration = essay_iterations[-1]
            final_essays.append({
                'Question': final_iteration['question'],
                'Essay': final_iteration['essay'], 
                'Overall': final_iteration.get('target_band', final_iteration['scores'].get('overall', 'N/A'))
            })
    if final_essays:
        final_df = pd.DataFrame(final_essays)
        combined_filename = f'band_{band_str}_{timestamp}.csv'
        final_df.to_csv(combined_filename, index=False)
        print(f"\nSaved {len(final_essays)} essays from bands {', '.join(map(str, args.bands))} to {combined_filename}")
    print_combined_summary_statistics(all_results, args)

def print_combined_summary_statistics(all_results: List[Dict], args: argparse.Namespace) -> None:
    """Print summary statistics for combined results from all bands."""
    print(f"\nCOMBINED SUMMARY RESULTS:")
    print(f"Target bands: {', '.join(map(str, args.bands))}")
    print(f"Total essays processed: {args.num_essays}")
    band_results = {}
    for band in args.bands:
        band_results[band] = []
    for essay_num in range(1, args.num_essays + 1):
        essay_iterations = [r for r in all_results if r['essay_number'] == essay_num]
        if essay_iterations:
            final_iteration = essay_iterations[-1]
            target_band = final_iteration.get('target_band')
            final_score = final_iteration['scores'].get('overall', 0)
            if target_band and target_band in band_results:
                band_results[target_band].append({
                    'essay_num': essay_num,
                    'final_score': final_score,
                    'target_reached': final_score >= target_band
                })
    total_success = 0
    total_essays = 0
    for band in sorted(args.bands):
        essays = band_results[band]
        if essays:
            success_count = sum(1 for e in essays if e['target_reached'])
            avg_score = sum(e['final_score'] for e in essays) / len(essays)
            print(f"\nBand {band} Results:")
            print(f"  Essays: {len(essays)}")
            print(f"  Success rate: {success_count}/{len(essays)} ({success_count/len(essays)*100:.1f}%)")
            print(f"  Average final score: {avg_score:.1f}")
            total_success += success_count
            total_essays += len(essays)
    if total_essays > 0:
        overall_success_rate = total_success / total_essays * 100
        print(f"\nOVERALL SUCCESS RATE: {total_success}/{total_essays} ({overall_success_rate:.1f}%)")

# MAIN FUNCTION

def parse_bands(band_string: str) -> List[int]:
    """Parse comma-separated band values and validate them."""
    try:
        bands = [int(b.strip()) for b in band_string.split(',')]
        for band in bands:
            if band not in range(5, 10):
                raise ValueError(f"Band {band} is not in valid range 5-9")
        return bands
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid band specification: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate synthetic IELTS essays using Gemini')
    parser.add_argument('bands', type=parse_bands, 
                       help='Target band score(s). Single band (e.g., 7) or comma-separated (e.g., 5,6,7)')
    parser.add_argument('--num-essays', type=int, default=1,
                       help='Number of essays to generate (default: 1)')
    parser.add_argument('--max-iterations', type=int, default=1,
                       help='Maximum iterations for improvement (default: 1)')
    args = parser.parse_args()
    # Check API key
    if not gemini_api_key:
        print("Error: Gemini API key required. Set GEMINI_API_KEY environment variable in .env file")
        sys.exit(1)
    # Load data
    fewshot_df, questions_df = load_data()
    print(f"Running iterative improvement mode:")
    print(f"- Target bands: {', '.join(map(str, args.bands))}")
    print(f"- Number of essays per band: {args.num_essays}")
    print(f"- Max iterations per essay: {args.max_iterations}")
    combined_results = []
    essay_counter = 1
    for band in args.bands:
        print(f"\n{'='*80}")
        print(f"PROCESSING TARGET BAND {band}")
        print(f"{'='*80}")
        for essay_num in range(args.num_essays):
            print(f"\n{'='*60}")
            print(f"PROCESSING ESSAY {essay_counter} (Band {band}, Essay {essay_num + 1}/{args.num_essays})")
            print(f"{'='*60}")
            question = questions_df.sample(n=1).iloc[0]['question']
            iterations = improve_essay_iteratively(
                question=question,
                target_band=band,
                max_iterations=args.max_iterations
            )
            if iterations:
                for iteration in iterations:
                    iteration['essay_number'] = essay_counter
                    iteration['target_band'] = band
                    iteration['question'] = question
                combined_results.extend(iterations)
            essay_counter += 1
    if combined_results:
        combined_args = argparse.Namespace()
        combined_args.bands = args.bands
        combined_args.num_essays = args.num_essays * len(args.bands)
        combined_args.max_iterations = args.max_iterations
        save_combined_results(combined_results, combined_args)

if __name__ == "__main__":
    main()
