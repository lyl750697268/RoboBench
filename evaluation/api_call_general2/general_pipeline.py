from contextlib import nullcontext
from prompt_utils import *
from api_utils import request_model
import asyncio
import argparse
import json, os

def main():
    parser = argparse.ArgumentParser(description="Request model completion for questions")
    parser.add_argument('--base-url', type=str, help="Base URL for the API")
    parser.add_argument('--api-key', type=str, help="API key for the API")
    parser.add_argument('--questions_file', type=str, help="Path to questions file")
    parser.add_argument('--output_file', type=str, help="Path to output file")
    parser.add_argument('--result-dir', type=str, help="Path to result directory")
    parser.add_argument('--result_file', type=str, help="Path to result file")
    parser.add_argument('--model', type=str, default="gpt-4o")
    parser.add_argument('--instruction_key', type=str, default=None)
    parser.add_argument('--question_key', type=str, default='question')
    parser.add_argument('--image_key', type=str, default=None)
    parser.add_argument('--system_prompt_file', type=str, default=None)
    parser.add_argument('--template', type=str, default=None)
    parser.add_argument('--mode', type=str, default='base64')
    args = parser.parse_args()
    base_url = args.base_url
    api_key = args.api_key
    questions_file = args.questions_file
    output_file = args.output_file
    result_file = args.result_file
    result_dir = args.result_dir
    model = args.model
    instruction_key = args.instruction_key
    question_key = args.question_key
    image_key = args.image_key
    system_prompt_file = args.system_prompt_file
    template = args.template
    mode = args.mode
    if output_file is not None:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    if template is None:
        prompt_no_template(questions_file, output_file, instruction_key, question_key, image_key, system_prompt_file)
    else:
        prompt_with_template(questions_file, output_file, template, image_key, system_prompt_file)
    if result_file is None:
        existing_results = None
    elif os.path.exists(result_file):
        with open(result_file, 'r') as f:
            results_list = json.load(f)
           
            valid_results = [result for result in results_list if result is not None]
            
            if len(valid_results) == 0:
                print(f"Warning: Result file exists but all results are null, will reprocess all requests")
                existing_results = None
            elif len(valid_results) < len(results_list) * 0.1:
                print(f"Warning: Result file has only {len(valid_results)}/{len(results_list)} (less than 10%) valid results, will reprocess all requests")
                existing_results = None
            else:
                existing_results = {result['id']: result for result in valid_results}
                null_count = len(results_list) - len(valid_results)
                if null_count > 0:
                    print(f"Read {len(existing_results)} valid results from result file (filtered out {null_count} null values)")
                else:
                    print(f"Read {len(existing_results)} results from result file")
    else:
        existing_results = None
    prompts = []
    results = []
    with open(output_file, 'r') as f:
        for line in f:
            prompt = json.loads(line)
            if existing_results is not None and prompt['request_id'] in existing_results:
                if existing_results[prompt['request_id']]['response'] is None:
                    prompts.append(prompt)
                else:
                    results.append(existing_results[prompt['request_id']])
            else:
                prompts.append(prompt)
        prompt = prompts
        if len(prompt) == 0:
            return
        messages, request_ids = format_messages(prompt, mode)
        print(f"Processing {len(messages)} requests...")
        print(messages[0])
        
        try:
            total_timeout = max(2000, len(messages) * 2)
            print(f"Set total timeout: {total_timeout} seconds")
            
            async def run_with_timeout():
                return await asyncio.wait_for(
                    request_model(messages, request_ids, api_model=model, base_url=base_url, api_key=api_key),
                    timeout=total_timeout
                )
            
            results += asyncio.run(run_with_timeout())
        except asyncio.TimeoutError:
            print(f"Batch processing timed out ({total_timeout} seconds), trying to read temporary results...")
            temp_file = "results_temp_result.json"
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"Read {len([r for r in results if r is not None])} results from temporary file")
            else:
                print("Temporary result file not found, returning empty results")
                results += [None] * len(messages)
        except Exception as e:
            print(f"Error during batch processing: {type(e).__name__} - {str(e)}")
            temp_file = "results_temp_result.json"
            if os.path.exists(temp_file):
                with open(temp_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                print(f"Read {len([r for r in results if r is not None])} results from temporary file")
            else:
                print("Temporary result file not found, returning empty results")
                results += [None] * len(messages)
    result_dir = os.path.join(result_dir, os.path.dirname(questions_file).split('/')[-1])
    os.makedirs(result_dir, exist_ok=True)
    if args.image_key is not None:
        anno_result_file = os.path.join(result_dir, f"{model}_result_{os.path.basename(questions_file)}")
    else:
        anno_result_file = os.path.join(result_dir, f"{model}_lang_result_{os.path.basename(questions_file)}")
    print(f"anno_result_file : {anno_result_file}")
    with open(anno_result_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()