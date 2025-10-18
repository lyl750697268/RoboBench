import json
import os
import re
import argparse
from typing import Dict, List, Any, Tuple, Union
from openai import OpenAI

def normalize_response_multiple_choice(answer: str, openai_client) -> str:
    """Normalize multiple choice answers by extracting the most accurate answer using GPT."""
    matches = re.findall(r'[A-D]', answer.upper())
    if len(matches) == 1:
        return ','.join(matches)
    elif len(matches) > 1:
        prompt = f"""
        Extract ONLY the final multiple choice answer(s) from this response. 
        Return ONLY the letter(s) A,B,C,D with no other text or explanation.
        If multiple answers, separate with commas.
        Response: {answer}
        """
        try:
            completion = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=80 
            )
            answer = completion.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error extracting answer with GPT: {e}")
            matches = re.findall(r'[A-D]', answer.upper())
            if matches:
                answer = ','.join(matches)
        return answer.strip().upper()
    else:
        return answer.strip().upper()
    
def normalize_gt_multiple_choice(answer: str) -> str:
    """Normalize multiple choice answers by extracting the most accurate answer."""
    matches = re.findall(r'[A-D]', answer.upper())
    if not matches:
        return answer.strip().upper()
    return ''.join(sorted(matches))
    

def evaluate_response(response: str, gt_answer: str, openai_client = None) -> Tuple[float, str]:
    """
    Evaluate a single response against the ground truth answer.
    Returns a score (0 or 1 for multiple choice) and an explanation.
    """
    normalized_response = normalize_response_multiple_choice(response, openai_client)
    normalized_gt = normalize_gt_multiple_choice(gt_answer)
    
    is_correct = normalized_response == normalized_gt
    score = 1.0 if is_correct else 0.0
    explanation = f"Response '{normalized_response}' {'matches' if is_correct else 'does not match'} ground truth '{normalized_gt}'"
    
    return score, explanation

def evaluate_results_file(file_path: str, openai_api_key: str = None, base_url: str = None) -> Dict[str, Any]:
    """
    Evaluate all responses in a results file.
    Returns a dictionary with evaluation statistics.
    """
    with open(file_path, 'r') as f:
        results = json.load(f)
    
    # Initialize OpenAI client if API key is provided
    openai_client = None
    if openai_api_key:
        openai_client = OpenAI(base_url=base_url, api_key=openai_api_key)
    
    total_questions = len(results)
    correct_answers = 0
    evaluation_details = []
    
    for item in results:
        if item is None:
            continue
        response = item.get('response', '')
        gt_answer = item.get('gt_answer', '')
        
        # Skip items without response or ground truth
        if not response or not gt_answer:
            continue
        
        # Evaluate the response
        try:
            score, explanation = evaluate_response(
                response, gt_answer, openai_client
            )
            
            is_correct = score >= 1.0
            if is_correct:
                correct_answers += 1
            
            # Store evaluation details
            evaluation_details.append({
                'id': item.get('id', ''),
                'response': response,
                'gt_answer': gt_answer,
                'score': score,
                'explanation': explanation,
                'is_correct': is_correct
            })
            
        except Exception as e:
            print(f"Error evaluating item {item.get('id', '')}: {e}")
    
    # Calculate accuracy
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    # Prepare evaluation summary
    evaluation_summary = {
        'total_questions': total_questions,
        'correct_answers': correct_answers,
        'overall_accuracy': accuracy,
        'evaluation_details': evaluation_details
    }
    
    return evaluation_summary

def save_evaluation_results(evaluation_summary: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    
    print(f"Evaluation results saved to {output_path}")
    print(f"Overall accuracy: {evaluation_summary['overall_accuracy']:.2%}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate GPT responses against ground truth answers')
    parser.add_argument('--results_file', type=str, help='Path to the results file containing responses and ground truth answers')
    parser.add_argument('--output', type=str, default='evaluation_results.json', help='Path to save evaluation results')
    parser.add_argument('--openai-api-key', type=str, default='Your own OpenAI API key', help='OpenAI API key for evaluating open-ended questions')
    parser.add_argument('--base-url', type=str, default='https://api.openai.com/v1', help='Base URL for the OpenAI API')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Evaluate responses
    evaluation_summary = evaluate_results_file(args.results_file, args.openai_api_key, args.base_url)
    
    # Save evaluation results
    save_evaluation_results(evaluation_summary, args.output)

if __name__ == "__main__":
    main() 