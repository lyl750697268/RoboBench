import requests
import base64
import os
import re
import json
import ast
from tqdm import tqdm
import asyncio
from glob import glob
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as async_tqdm
from datetime import datetime
import base64
import mimetypes
from pathlib import Path
import shutil


# Import prompts from unified prompt module
from unified_prompts import (
    PROMPT_TEMPLATE_EXTRACT_PLAN_INFO, 
    PROMPT_TEMPLATE_Q1_EVALUATION,
    PROMPT_TEMPLATE_EXTRACT_STEP,
    PROMPT_TEMPLATE_COMPARE_STEPS,
    PROMPT_TEMPLATE_EXTRACT_YES_NO
)

# API configuration
base_url = "Your own base URL for the API"
api_key = "Your own OpenAI API key"

# Set concurrent limit
MAX_CONCURRENT = 50
EVAL_MODEL = "gpt-4o-2024-11-20"

# Global variables, set in main function
eval_data_dir = None
intermediate_results_dir = None

def setup_directories(base_dir):
    """Set up subdirectories for evaluation required for specified directory
    
    Args:
        base_dir: Directory path to process
        
    Returns:
        tuple: (intermediate_results_dir, result_dir)
    """
    # Create intermediate result save directory
    intermediate_results_dir = os.path.join(base_dir, "intermediate_results")
    os.makedirs(intermediate_results_dir, exist_ok=True)
    
    # Create result directory
    result_dir = os.path.join(base_dir, "result")
    os.makedirs(result_dir, exist_ok=True)
    
    # Create subdirectories for Q1, Q2, Q3
    for q_type in ['q1', 'q2', 'q3']:
        # Intermediate result subdirectory
        subdir = os.path.join(intermediate_results_dir, q_type)
        os.makedirs(subdir, exist_ok=True)
        
        # Result subdirectory
        result_subdir = os.path.join(result_dir, q_type)
        os.makedirs(result_subdir, exist_ok=True)
        
    return intermediate_results_dir, result_dir

def move_outputs_to_version(base_dir, version):
    """Move result, intermediate_results, and result.txt from base_dir to version subdirectory.

    Args:
        base_dir: Directory path
        version: Version subdirectory name
    """
    try:
        if not version:
            return
        version_dir = os.path.join(base_dir, version)
        os.makedirs(version_dir, exist_ok=True)

        # Move directories result and intermediate_results
        for name in ["result", "intermediate_results"]:
            src = os.path.join(base_dir, name)
            dst = os.path.join(version_dir, name)
            if os.path.exists(src):
                try:
                    if os.path.exists(dst):
                        # If target exists, delete it first to replace
                        shutil.rmtree(dst)
                    shutil.move(src, dst)
                    print(f"Moved {src} -> {dst}")
                except Exception as e:
                    print(f"Failed to move directory {name}: {e}")

        # Move file result.txt
        src_file = os.path.join(base_dir, "result.txt")
        dst_file = os.path.join(version_dir, "result.txt")
        if os.path.exists(src_file):
            try:
                if os.path.exists(dst_file):
                    os.replace(src_file, dst_file)
                else:
                    shutil.move(src_file, dst_file)
                print(f"Moved {src_file} -> {dst_file}")
            except Exception as e:
                print(f"Failed to move file result.txt: {e}")
    except Exception as e:
        print(f"Error moving to version directory: {e}")

# Utility functions
def extract_dict_string(data_string):
    """
    Extract dictionary format string that can be converted by ast.literal_eval from input string

    Args:
    data_string (str): String containing dictionary data

    Returns:
    str: Extracted dictionary format string
    """
    pattern = re.compile(r'\{.*\}', re.DOTALL)
    match = pattern.search(data_string)
    if match:
        return match.group()  # Extracted dictionary string
    else:
        print("No valid dictionary data found")
        raise ValueError("No valid dictionary data found")

def save_list_of_dicts_to_json(data_list, file_name):
    """
    Merge and save list data containing dictionaries with existing JSON file data

    Args:
    data_list (list): New data list containing dictionaries
    file_name (str): Output JSON file name (including path)
    """
    try:
        # Read original file data
        try:
            with open(file_name, 'r', encoding='utf-8') as json_file:
                existing_data = json.load(json_file)
                if not isinstance(existing_data, list):
                    raise ValueError("Existing data in file is not list format")
        except FileNotFoundError:
            existing_data = []
        except json.JSONDecodeError:
            existing_data = []
        
        # Merge data
        combined_data = existing_data + data_list

        # Write back to file
        with open(file_name, 'w', encoding='utf-8') as json_file:
            json.dump(combined_data, json_file, ensure_ascii=False, indent=4)
        print(f"Data successfully saved to {file_name}")
    except Exception as e:
        print(f"Error saving data: {e}")

def save_intermediate_result(data, step_name, timestamp=None):
    """
    Save intermediate result to specified file

    Args:
    data: Data to save
    step_name (str): Step name
    timestamp (str): Timestamp, if None, generate automatically
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"{step_name}_{timestamp}.json"
    filepath = os.path.join(intermediate_results_dir, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Intermediate result saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Failed to save intermediate result: {e}")
        return None

def load_intermediate_result(step_name, timestamp=None):
    """
    Load intermediate result

    Args:
    step_name (str): Step name
    timestamp (str): Timestamp, if None, load the latest

    Returns:
    Loaded data or None
    """
    try:
        if timestamp:
            filename = f"{step_name}_{timestamp}.json"
            filepath = os.path.join(intermediate_results_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        else:
            # Find the latest file
            pattern = f"{step_name}_*.json"
            files = glob(os.path.join(intermediate_results_dir, pattern))
            if files:
                latest_file = max(files, key=os.path.getctime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    print(f"Loaded intermediate result: {latest_file}")
                    return json.load(f)
    except Exception as e:
        print(f"Failed to load intermediate result: {e}")
    return None

def remove_reason(data):
    temp = {}
    temp["plan_step"] = data["plan_step"]
    return temp

def determine_planning_question_type(gt_answer: str) -> str:
    """
    Determine the type of planning question based on gt_answer
    - q3: yes/no type
    - q2: Only one function
    - q1: Multiple functions
    
    Args:
        gt_answer: Ground truth answer
    
    Returns:
        'q1', 'q2', or 'q3'
    """
    import re
    
    # Check if it is yes/no type
    yes_no_keywords = ['yes', 'no', 'true', 'false']
    if gt_answer.lower().strip() in yes_no_keywords:
        return 'q3'
    
    # Check if it contains multiple steps (multiple function calls)
    # Find patterns like skill(element1, element2)
    function_pattern = r'\w+\([^)]+\)'
    functions = re.findall(function_pattern, gt_answer)
    
    if len(functions) > 1:
        return 'q1'  # Multiple functions
    elif len(functions) == 1:
        return 'q2'  # Only one function
    else:
        # If the function pattern is not found, it may be other formats, temporarily classified as q2
        return 'q2'

def retry_error_callback(retry_state):
    exception = retry_state.outcome.exception()
    print(f"Retry attempt {retry_state.attempt_number} failed: {type(exception).__name__} - {str(exception)}")
    return None

async def save_temp_results(results, save_path, current_count):
    temp_file = f"{save_path}_temp_result.json"
    last_file = f"{save_path}_temp_result_last.json"
    
    # If the old file exists, rename it to last
    if os.path.exists(temp_file):
        try:
            os.replace(temp_file, last_file)
        except Exception as e:
            print(f"Failed to rename old temp file: {e}")
    
    # Save new temporary file
    async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(results, ensure_ascii=False, indent=2))

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10), retry_error_callback=retry_error_callback)
async def get_chat_completion(messages: dict, request_id, semaphore, api_model=EVAL_MODEL, base_url=base_url, api_key=api_key, retry_count=0) -> str:
    response = None
    resp = {'id': request_id}
    try:
        async with semaphore:  # Use the incoming semaphore to limit concurrency
            response = await AsyncOpenAI(
                base_url=base_url, api_key=api_key, timeout=360
            ).chat.completions.create(
                model=api_model, messages=messages, timeout=360, temperature=0  # Set 60 seconds timeout
            )
            
            response_result = response.choices[0].message.content
            resp['response'] = response_result
            resp['model'] = api_model
            return resp
    except Exception as e:
        print(f"Error in get_chat_completion for request_id {request_id}: {type(e).__name__} - {str(e)}")
        if response:
            print(f"Response status: {response}")
        raise

async def request_model(prompts, request_ids, api_model=EVAL_MODEL, base_url=base_url, api_key=api_key, save_path="results"):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results = [None] * len(prompts)
    completed_count = 0
    failed_count = 0
    
    # Task level timeout (seconds)
    TASK_TIMEOUT = 960

    async def wrapped_get_chat_completion(prompt, request_id, index, api_model=EVAL_MODEL, base_url=base_url, api_key=api_key):
        nonlocal completed_count, failed_count
        try:
            # Add timeout protection for each task
            result = await asyncio.wait_for(
                get_chat_completion(prompt, request_id, semaphore, api_model, base_url, api_key),
                timeout=TASK_TIMEOUT
            )
            completed_count += 1
            
            # Save temporary result every 100 completed tasks
            if completed_count % 100 == 0:
                print(f"Completed {completed_count}/{len(prompts)} tasks")
                await save_temp_results([r for r in results if r is not None], save_path, completed_count)
                
            return index, result
        except asyncio.TimeoutError:
            failed_count += 1
            print(f"Task {request_id} (index {index}) timed out after {TASK_TIMEOUT} seconds")
            return index, {'id': request_id, 'response': None, 'error': 'timeout', 'model': api_model}
        except Exception as e:
            failed_count += 1
            print(f"Task {request_id} (index {index}) failed after all retries: {type(e).__name__} - {str(e)}")
            return index, {'id': request_id, 'response': None, 'error': str(e), 'model': api_model}

    # Create all tasks
    tasks = [wrapped_get_chat_completion(prompt, request_ids[i], i, api_model) for i, prompt in enumerate(prompts)]
    
    print(f"Start processing {len(tasks)} tasks, maximum concurrent number: {MAX_CONCURRENT}")
    
    # Use asyncio.as_completed instead of tqdm.as_completed, avoid dependency issues
    completed_tasks = 0
    for future in asyncio.as_completed(tasks):
        try:
            index, result = await future
            results[index] = result
            completed_tasks += 1
            
            # Print progress every 50 completed tasks
            if completed_tasks % 50 == 0 or completed_tasks == len(tasks):
                print(f"Progress: {completed_tasks}/{len(tasks)} ({completed_tasks/len(tasks)*100:.1f}%) - Success: {completed_count}, Failed: {failed_count}")
                
        except Exception as e:
            print(f"Unexpected error in task completion: {e}")
            continue

    # Save final result
    print(f"All tasks completed! Success: {completed_count}, Failed: {failed_count}")
    await save_temp_results(results, save_path, "final")
    return results



async def process_model_requests_in_memory(task_list, model_name=EVAL_MODEL, base_url=base_url, api_key=api_key):
    """
    Process model requests in memory using async API calls

    Args:
        task_list (list): Each task has id, prompt, image_urls(local path list)
        model_name (str): Model name (need to support multi-modal/vision)
    """
    if not task_list:
        return []

    prompts = []
    request_ids = []

    for task in task_list:
        # Put text first
        user_content = [
            {"type": "text", "text": task["prompt"]}
        ]

        # Convert local images to data URL and append
        for p in (task.get("image_urls") or []):
            try:
                mime, _ = mimetypes.guess_type(p)
                if not mime:
                    mime = "image/png"
                b64 = base64.b64encode(Path(p).read_bytes()).decode("utf-8")
                data_url = f"data:{mime};base64,{b64}"

                # Most SDKs accept this image_url block; if you need to use input_image, change type to "input_image"
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })
            except Exception as e:
                # Don't interrupt the entire batch: when an error occurs, put the error message in the text (can be removed as needed)
                user_content.append({
                    "type": "text",
                    "text": f"[Image load error for {p}: {e}]"
                })

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are an AI assistant that helps people find information."}
                ]
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        prompts.append(messages)
        request_ids.append(task["id"])

    # Use asynchronous batch processing
    results = await request_model(prompts, request_ids, api_model=model_name, base_url=base_url, api_key=api_key, save_path="temp_results")
    return results


# Step 1: Classify and Extract Plan Information for Q1, Q2, Q3
async def classify_and_extract_planning_data(use_cache=True):
    """Classify and extract Q1, Q2, Q3 plan information"""
    print("Starting classify_and_extract_planning_data step...")
    
    # Try to load cached intermediate result
    if use_cache:
        cached_result = load_intermediate_result("classified_planning_data")
        if cached_result:
            print("Found cached classified_planning_data results, directly use")
            return cached_result
    
    # Find json files in current directory
    file_list = glob(os.path.join(eval_data_dir, "*.json"))
    print(f"Found files: {file_list}")
    
    # Classify store
    all_classified_data = {
    'q1': {},  # Store Q1 tasks
    'q2': {},  # Store Q2 tasks
    'q3': {}   # Store Q3 tasks
    }
    
    for file_path in file_list:
        with open(file_path, 'r') as file:
            data = json.load(file)
        
        file_key = file_path.split('/')[-1]
        print(f"Process file: {file_key}, contains {len(data)} records")
        
        # Classify data by question type
        q1_data = []
        q2_data = []
        q3_data = []
        
        for i in tqdm(range(len(data)), desc=f"Classify {file_key}"):
            try:
                record = data[i]
                task_id = record["id"]
                
                # Determine question type
                if "_Q1" in task_id:
                    q_type = 'q1'
                    q1_data.append(record)
                elif "_Q2" in task_id:
                    q_type = 'q2'
                    q2_data.append(record)
                elif "_Q3" in task_id:
                    q_type = 'q3'
                    q3_data.append(record)
                else:
                    # Automatically determine question type based on gt_answer
                    gt_answer = record.get("gt_answer", "")
                    q_type = determine_planning_question_type(gt_answer)
                    if q_type == 'q1':
                        q1_data.append(record)
                    elif q_type == 'q2':
                        q2_data.append(record)
                    else:
                        q3_data.append(record)
                
            except Exception as e:
                print(f"Error processing record {i}: {e}")
                continue
        
        # Store classified result
        all_classified_data['q1'][file_key] = q1_data
        all_classified_data['q2'][file_key] = q2_data
        all_classified_data['q3'][file_key] = q3_data
        
        print(f"  Q1: {len(q1_data)} records")
        print(f"  Q2: {len(q2_data)} records")
        print(f"  Q3: {len(q3_data)} records")
    
    # Save intermediate result
    save_intermediate_result(all_classified_data, "classified_planning_data")
    
    return all_classified_data

# Step 1.1: Extract Plan Information for Q1 tasks only
async def extract_plan_information_q1(classified_data, use_cache=True, base_url=base_url, api_key=api_key):
    """Only process Q1 task plan information extraction"""
    print("Starting extract_plan_information_q1 step...")
    
    if use_cache:
        cached_result = load_intermediate_result("extract_plan_information_q1")
        if cached_result:
            print("Found cached extract_plan_information_q1 results, directly use")
            return cached_result
    
    all_extracted_data = {}
    q1_data = classified_data['q1']
    
    for file_key, data in q1_data.items():
        if not data:
            continue
            
        print(f"Process Q1 file: {file_key}, contains {len(data)} records")
        
        # Prepare tasks
        task_model_plan = []
        
        # Check processed data
        try:
            processed_file = f"{eval_data_dir}/result/q1/extract_plan_information_{file_key}"
            with open(processed_file, 'r') as file1:
                processed_data = json.load(file1)
        except:
            processed_data = []
        
        have_processed_id = [temp["id"] for temp in processed_data]
        
        for record in data:
            task_id = record["id"]
            if task_id in have_processed_id:
                continue
            
            record["previous_step"] = ""
            response_data = {
                "prompt": record["question"],
                "previous_step": record["previous_step"],
                "response": record["response"]
            }
            
            # Use unified prompt template
            response_prompt = PROMPT_TEMPLATE_EXTRACT_PLAN_INFO.format(data=str(response_data))
            task_model_plan.append({"id": task_id, "prompt": response_prompt, "image_urls": []})
        
        # Process all tasks
        if len(task_model_plan) > 0:
            print(f"Start processing {len(task_model_plan)} Q1 tasks...")
            model_results = await process_model_requests_in_memory(task_model_plan, base_url=base_url, api_key=api_key)
        else:
            model_results = []
            
        # Process result
        final_result = []
        for idx in range(len(model_results)):
            if model_results[idx] is not None:
                task_id = model_results[idx]["id"]
            else:
                continue
            
            # Find corresponding original data
            record = None
            for r in data:
                if r["id"] == task_id:
                    record = r
                    break
            
            if record is None:
                continue

            record["previous_step"] = ""
            # Use regex to match comma, ensure it splits after numbers
            # split_list = re.split(r',\s*(?=\d+-)', f'{record["previous_step"]}, {record["gt_answer"]}')
            split_list = re.split(r',\s*(?=\d+-)', f'{record["gt_answer"]}')
            previous_list = re.split(r',\s*(?=\d+-)', f'{record["previous_step"]}')

            for split_idx in range(len(split_list)):
                split_list[split_idx] = split_list[split_idx][2:].replace("-", "")
            for split_idx in range(len(previous_list)):
                previous_list[split_idx] = previous_list[split_idx][2:]
            
            try:
                extracted_dict = ast.literal_eval(extract_dict_string(model_results[idx]["response"]))
                final_result.append({
                    "id": task_id, 
                    "model_plan": extracted_dict, 
                    "gt_plan": {
                        'task_summary': extracted_dict.get('task_summary', ''), 
                        'plan_step': split_list, 
                        "known_plan_step": previous_list
                    }, 
                    "model": model_results[idx]["model"], 
                    "known_plan_step": previous_list
                })
            except Exception as e:
                print(f"Q1 task {task_id} result format is incorrect: {e}")
                continue
        
        # Save Q1 results
        save_list_of_dicts_to_json(final_result, f"{eval_data_dir}/result/q1/extract_plan_information_{file_key}")
        all_extracted_data[file_key] = final_result
    
    # Save intermediate results
    save_intermediate_result(all_extracted_data, "extract_plan_information_q1")
    
    return all_extracted_data

# Step 1.2: Process Q2 tasks (step extraction and comparison)
async def process_q2_tasks(classified_data, use_cache=True, base_url=base_url, api_key=api_key):
    """Process Q2 tasks (step extraction and comparison)"""
    print("Starting process_q2_tasks step...")
    
    if use_cache:
        cached_result = load_intermediate_result("process_q2_tasks")
        if cached_result:
            print("Found cached process_q2_tasks results, directly use")
            return cached_result
    
    all_q2_results = {}
    q2_data = classified_data['q2']
    
    for file_key, data in q2_data.items():
        if not data:
            continue
            
        print(f"Process Q2 file: {file_key}, contains {len(data)} records")
        
        # Step 1: Batch extract steps
        extract_prompts = []
        extract_request_ids = []
        
        for record in data:
            task_id = record["id"]
            pred_answer = record["response"]
            
            # Use LLM to extract step
            prompt = PROMPT_TEMPLATE_EXTRACT_STEP.format(response=pred_answer)
            extract_prompts.append(prompt)
            extract_request_ids.append(f"{task_id}_extract")
        
        # Batch execute step extraction
        if extract_prompts:
            print(f"Start batch executing Q2 step extraction, {len(extract_prompts)} tasks")
            extract_results = await process_model_requests_in_memory([
                {"id": rid, "prompt": prompt, "image_urls": []} 
                for prompt, rid in zip(extract_prompts, extract_request_ids)
            ], base_url=base_url, api_key=api_key)
        else:
            extract_results = []
        
        # Step 2: Prepare compare requests
        compare_prompts = []
        compare_request_ids = []
        valid_tasks = []  # Store extracted successful tasks
        
        for i, record in enumerate(data):
            task_id = record["id"]
            gt_answer = record["gt_answer"]
            
            # Check corresponding extraction result
            extract_result = extract_results[i] if i < len(extract_results) else None
            if not extract_result or not extract_result.get('response'):
                print(f"Task {task_id} step extraction failed, skip")
                continue
            
            extracted_step = extract_result['response'].strip()
            if not extracted_step:
                print(f"Task {task_id} extracted step is empty, skip")
                continue
            
            # Prepare compare prompt
            prompt = PROMPT_TEMPLATE_COMPARE_STEPS.format(
                extracted_step=extracted_step,
                gt_step=gt_answer.split("-")[-1]
            )
            compare_prompts.append(prompt)
            compare_request_ids.append(f"{task_id}_compare")
            
            # Save valid task information
            valid_tasks.append({
                'task_id': task_id,
                'extracted_step': extracted_step,
                'gt_answer': gt_answer.split("-")[-1]
            })
        
        # Step 3: Batch execute step comparison
        if compare_prompts:
            print(f"Start batch executing Q2 step comparison, {len(compare_prompts)} tasks")
            compare_results = await process_model_requests_in_memory([
                {"id": rid, "prompt": prompt, "image_urls": []} 
                for prompt, rid in zip(compare_prompts, compare_request_ids)
            ], base_url=base_url, api_key=api_key)
        else:
            compare_results = []
        
        # Step 4: Process final result
        final_result = []
        
        for i, valid_task in enumerate(valid_tasks):
            task_id = valid_task['task_id']
            
            # Check corresponding comparison result
            compare_result = compare_results[i] if i < len(compare_results) else None
            if not compare_result or not compare_result.get('response'):
                print(f"Task {task_id} step comparison failed");
                continue
            
            try:
                # Parse comparison result
                comparison_response = compare_result['response'].strip()
                comparison_result_dict = ast.literal_eval(extract_dict_string(comparison_response))
                
                # Extract three scores
                skill_accuracy = comparison_result_dict.get('skill_usage_accuracy', {}).get('result', 0)
                object_reasonableness = comparison_result_dict.get('operation_object_reasonableness', {}).get('result', 0)
                parameter_accuracy = comparison_result_dict.get('parameter_accuracy', {}).get('result', 0)
                
                # Record reason
                skill_reason = comparison_result_dict.get('skill_usage_accuracy', {}).get('reason', 'No reason provided')
                object_reason = comparison_result_dict.get('operation_object_reasonableness', {}).get('reason', 'No reason provided')
                parameter_reason = comparison_result_dict.get('parameter_accuracy', {}).get('reason', 'No reason provided')
                
                # Calculate final score based on new scoring logic
                # Skill: 0 or 1 score; Object: 0, 0.5 or 1 score; Parameter: 0, 0.5 or 1 score (if skill or object is 0, parameter is automatically 0)
                if skill_accuracy == 0 or object_reasonableness == 0:
                    parameter_accuracy = 0  # If skill or object is 0, parameter is 0
                
                # Calculate final score: weighted average of three parts, converted to 100 point system
                final_score = (skill_accuracy + object_reasonableness + parameter_accuracy) / 3.0 * 100.0
                final_score = min(final_score, 100.0)
                final_score = max(0.0, final_score)
                final_score = round(final_score, 2)
                
                final_result.append({
                    'task_id': task_id, 
                    'score': final_score,
                    'details': {
                        'extracted_step': valid_task['extracted_step'],
                        'gt_step': valid_task['gt_answer'],
                        'skill_accuracy': skill_accuracy,
                        'object_reasonableness': object_reasonableness,
                        'parameter_accuracy': parameter_accuracy,
                        'total_score': final_score,
                        'skill_reason': skill_reason,
                        'object_reason': object_reason,
                        'parameter_reason': parameter_reason
                    }
                })
                
            except Exception as e:
                print(f"Task {task_id} Q2 score calculation failed: {e}")
                continue
        
        # Save Q2 results
        save_list_of_dicts_to_json(final_result, f"{eval_data_dir}/result/q2/process_q2_{file_key}")
        all_q2_results[file_key] = final_result
    
    # Save intermediate results
    save_intermediate_result(all_q2_results, "process_q2_tasks")
    
    return all_q2_results

# Step 1.3: Process Q3 tasks (yes/no extraction and comparison)
async def process_q3_tasks(classified_data, use_cache=True, base_url=base_url, api_key=api_key):
    """Process Q3 tasks (yes/no extraction and comparison)"""
    
    if use_cache:
        cached_result = load_intermediate_result("process_q3_tasks")
        if cached_result:
            print("Found cached process_q3_tasks results, directly use")
            return cached_result
    
    all_q3_results = {}
    q3_data = classified_data['q3']
    
    for file_key, data in q3_data.items():
        if not data:
            continue
            
        print(f"Process Q3 file: {file_key}, contains {len(data)} records")
        
        prompts = []
        request_ids = []
        metas = []
        
        for record in data:
            task_id = record["id"]
            pred_answer = record["response"]
            gt_answer = record["gt_answer"].lower().strip()
            
            prompts.append(PROMPT_TEMPLATE_EXTRACT_YES_NO.format(response=pred_answer))
            request_ids.append(f"{task_id}_extract")
            metas.append((task_id, pred_answer, gt_answer))
        
        # Batch execute yes/no extraction
        if prompts:
            extract_results = await process_model_requests_in_memory([
                {"id": rid, "prompt": prompt, "image_urls": []} 
                for prompt, rid in zip(prompts, request_ids)
            ], base_url=base_url, api_key=api_key)
        else:
            extract_results = []
        
        final_result = []
        for idx, (task_id, pred_answer, gt_answer) in enumerate(metas):
            try:
                result = extract_results[idx] if idx < len(extract_results) else None
                if result and result.get('response'):
                    extracted_answer = result['response'].strip().lower()
                    is_correct = extracted_answer == gt_answer
                    score = 100.0 if is_correct else 0.0
                    method = 'llm_extract'
                else:
                    extracted_answer = ''
                    score = 0.0
                    method = 'llm_failed'
                
                # 保留两位小数
                score = round(score, 2)
                
                final_result.append({
                    'task_id': task_id,
                    'score': score,
                    'details': {
                        'pred_answer': pred_answer,
                        'gt_answer': gt_answer,
                        'extracted_answer': extracted_answer,
                        'method': method
                    }
                })
            except Exception as e:
                print(f"Task {task_id} Q3 score calculation failed: {e}")
                final_result.append({'task_id': task_id, 'score': 0.0, 'error': str(e)})
        
        # Save Q3 results
        save_list_of_dicts_to_json(final_result, f"{eval_data_dir}/result/q3/process_q3_{file_key}")
        all_q3_results[file_key] = final_result
    
    # Save intermediate results
    save_intermediate_result(all_q3_results, "process_q3_tasks")
    
    return all_q3_results

# Step 2: DAG Generation for Q1 tasks only
async def dag_llm_evaluation_q1(extracted_data_q1=None, use_cache=True, dataset_base_dir=None, skip_missing_resources=True, base_url=base_url, api_key=api_key):
    """Generate DAG from extracted Q1 plan information
    
    Now read DAG information from local dag.json file for each dimension, instead of from the unified GT_DAG_all_data.json file
    
    Args:
        extracted_data_q1: Q1 extracted data
        use_cache: Whether to use cache
        dataset_base_dir: Dataset base directory
        skip_missing_resources: Whether to skip tasks with missing resources (default True)
    """
    print("Starting dag_llm_evaluation_q1 step...")
    
    # Try to load cached intermediate results
    if use_cache and extracted_data_q1 is None:
        cached_result = load_intermediate_result("dag_llm_evaluation_q1")
        if cached_result:
            print("Found cached dag_llm_evaluation_q1 results, directly use")
            return cached_result
    
    all_dag_data = {}  # Store all DAG generation results
    
    if extracted_data_q1 is not None:
        # Use data in memory
        data_sources = extracted_data_q1
    else:
        # Fall back to file reading
        file_list = glob(f"{eval_data_dir}/result/q1/extract_plan_information_*")
        print(f"Load Q1 extracted data from files: {file_list}")
        data_sources = {}
        for file_path in file_list:
            with open(file_path, 'r') as file:
                data_sources[file_path.split('extract_plan_information_')[-1]] = json.load(file)
    
    for file_key, data in data_sources.items():
        print(f"Process Q1 file: {file_key}, contains {len(data)} records")
        
        # Prepare DAG evaluation tasks
        task_model_plan = []
        found_gt_count = 0
        missing_dag_count = 0
        missing_image_count = 0
        skipped_tasks = []
        
        for i in tqdm(range(len(data)), desc=f"Prepare Q1 DAG tasks - {file_key}"):
            task_id = data[i]["id"]
            
            # Extract dimension path and DAG information from task_id
            gt_dag, dag_image_path = load_dag_for_task(task_id, dataset_base_dir)
            
            if gt_dag is None:
                missing_dag_count += 1
                print(f"Warning: ID {task_id} not found corresponding DAG information")
                if skip_missing_resources:
                    skipped_tasks.append({"id": task_id, "reason": "missing_dag_data"})
                    continue
            
            # Validate image paths
            image_urls = build_image_paths_new(task_id, dag_image_path, dataset_base_dir)
            
            # Check if images exist
            valid_images = []
            missing_images = []
            for img_path in image_urls:
                if os.path.exists(img_path):
                    valid_images.append(img_path)
                else:
                    missing_images.append(img_path)
            
            if missing_images:
                missing_image_count += 1
                print(f"Warning: ID {task_id} missing images: {missing_images}")
                if skip_missing_resources and len(valid_images) < 2:
                    # Q1 tasks need at least 2 images (scene + DAG)
                    skipped_tasks.append({
                        "id": task_id, 
                        "reason": "insufficient_images",
                        "missing_images": missing_images,
                        "valid_images": valid_images
                    })
                    continue
            
            found_gt_count += 1
            
            try:
                response_data = {
                    'GT action list': data[i]["gt_plan"]['plan_step'],
                    'GT dag': gt_dag,
                    'model plan action list': data[i]["model_plan"]['plan_step']
                }
            except Exception as e:
                print(f"Failed to build response_data for {task_id}: {e}")
                if skip_missing_resources:
                    skipped_tasks.append({"id": task_id, "reason": "response_data_error", "error": str(e)})
                    continue
            
            # Use unified prompt template
            response_prompt = PROMPT_TEMPLATE_Q1_EVALUATION.format(data=str(response_data).replace("_handle", "").replace("handle_of_", ""))
            
            task_model_plan.append({
                "id": task_id, 
                "prompt": response_prompt, 
                "image_urls": valid_images  # Only use valid images
            })
        
        print(f"Prepare {len(task_model_plan)} Q1 DAG tasks")
        print(f"  - Found GT DAG: {found_gt_count}")
        print(f"  - Missing DAG: {missing_dag_count}")
        print(f"  - Missing images: {missing_image_count}")
        if skipped_tasks:
            print(f"  - Skip tasks: {len(skipped_tasks)}")
            # Save skipped tasks list
            skipped_file = f"{eval_data_dir}/result/q1/skipped_tasks_{file_key}"
            with open(skipped_file, 'w', encoding='utf-8') as f:
                json.dump(skipped_tasks, f, ensure_ascii=False, indent=2)
            print(f"  - Skip tasks list saved to: {skipped_file}")
            
        # Process all tasks
        if len(task_model_plan) > 0:
            print(f"Start asynchronous processing {len(task_model_plan)} Q1 DAG evaluation tasks...")
            model_results = await process_model_requests_in_memory(task_model_plan, base_url=base_url, api_key=api_key)
        else:
            model_results = []
            
        # Process results
        final_result = []
        successful_evaluations = 0  
        
        for idx in range(len(model_results)):
            try:
                if not model_results[idx] or not model_results[idx].get("response"):
                    continue
                    
                task_id = model_results[idx]["id"]
            except:
                continue
            
            # Find corresponding data record
            known_plan_step = []
            use_model = "unknown"
            task_summary = ""
            
            for idx_know in range(len(data)):
                if task_id != data[idx_know]['id']:
                    continue
                else:
                    known_plan_step = data[idx_know]["known_plan_step"]
                    use_model = data[idx_know]["model"]
                    task_summary = data[idx_know]["gt_plan"]["task_summary"]
                    break
            
            try:
                if not model_results[idx]["response"]:
                    print(f"Task {task_id}: Q1 DAG evaluation response is empty")
                    continue
                    
                final_score = ast.literal_eval(extract_dict_string(model_results[idx]['response']))
                
                # Extract problem analysis information (if exists)
                issue_analysis = final_score.get('planning_issue_analysis', {})
                
                final_result.append({
                    "id": task_id, 
                    "final_score": final_score, 
                    "known_plan_step": known_plan_step, 
                    "model": use_model, 
                    "task_summary": task_summary,
                    "planning_issues": issue_analysis
                })
                successful_evaluations += 1
            except Exception as e:
                print(f"Task {task_id} Q1 DAG evaluation result format is incorrect: {e}")
        
        print(f"Successfully evaluated {successful_evaluations} Q1 tasks")
        
        # Save results to all_dag_data and file
        all_dag_data[file_key] = final_result
        save_list_of_dicts_to_json(final_result, f"{eval_data_dir}/result/q1/generate_dag_{file_key}")
    
    # Save intermediate results
    save_intermediate_result(all_dag_data, "dag_llm_evaluation_q1")
    
    return all_dag_data

# Global DAG cache to avoid reading the same dag.json file multiple times
_dag_cache = {}

def clear_dag_cache():
    """Clear DAG cache"""
    global _dag_cache
    _dag_cache = {}
    print("DAG cache cleared")

def load_dag_for_task(task_id, dataset_base_dir=None):
    """Extract dimension information from task_id and load DAG information from corresponding dag.json file
    
    Args:
        task_id: Task ID, format like "3_generalized_planning/cross_embodiment/dual_arm/images/xxx_Q1"
        dataset_base_dir: Dataset base directory
        
    Returns:
        tuple: (gt_dag, dag_image_path) if found, otherwise (None, None)
    """
    # Remove _Q1, _Q2, _Q3 identifier at the end
    clean_id = task_id
    if task_id.endswith('_Q1') or task_id.endswith('_Q2') or task_id.endswith('_Q3'):
        clean_id = task_id[:-3]
    
    # Parse task_id to get dimension path
    # For example: "3_generalized_planning/cross_embodiment/dual_arm/images/xxx"
    parts = clean_id.split('/')
    
    if len(parts) < 3:
        print(f"Warning: task_id format is incorrect: {task_id}")
        return None, None
    
    # Find the position of "images" or other separators
    dimension_parts = []
    for part in parts:
        if part == 'images' or part.startswith('img'):
            break
        dimension_parts.append(part)
    
    if len(dimension_parts) < 2:
        print(f"Warning: cannot extract dimension information from task_id: {task_id}")
        return None, None
    
    # Build dimension folder path
    dimension_path = os.path.join(dataset_base_dir, *dimension_parts)
    dag_json_path = os.path.join(dimension_path, "dag.json")
    
    # Check if dag.json exists
    if not os.path.exists(dag_json_path):
        print(f"Warning: DAG file does not exist: {dag_json_path}")
        return None, None
    
    # Check cache
    if dag_json_path not in _dag_cache:
        # Read dag.json
        try:
            with open(dag_json_path, 'r', encoding='utf-8') as f:
                dag_data = json.load(f)
            _dag_cache[dag_json_path] = dag_data
        except Exception as e:
            print(f"Error: failed to read DAG file {dag_json_path}: {e}")
            return None, None
    else:
        dag_data = _dag_cache[dag_json_path]
    
    # Find matching record in dag_data
    gt_dag = None
    dag_image_filename = None
    
    for record in dag_data:
        record_id = record.get('id', '')
        # Ignore _Q1/_Q2/_Q3 identifier at the end when matching
        if record_id == clean_id or record_id == task_id:
            gt_dag = record.get('gt_dag')
            dag_image_filename = record.get('v2')  # DAG image filename
            break
    
    if gt_dag is None:
        print(f"Warning: no matching record found in DAG file: {task_id}")
        return None, None
    
    # Build DAG image full path
    dag_image_path = None
    if dag_image_filename:
        dag_folder = os.path.join(dimension_path, "DAG")
        dag_image_path = os.path.join(dag_folder, dag_image_filename)
        
        # Check if image exists
        if not os.path.exists(dag_image_path):
            print(f"Warning: DAG image does not exist: {dag_image_path}")
            dag_image_path = None
    
    return gt_dag, dag_image_path

def build_image_paths_new(task_id, dag_image_path, dataset_base_dir=None):
    """Build image path list based on task ID and DAG image path
    
    Args:
        task_id: Task ID
        dag_image_path: DAG image path
        dataset_base_dir: Dataset base directory
        
    Returns:
        list: Image path list [scene image, DAG image]
    """
    image_paths = []
    
    # Remove _Q1, _Q2, _Q3 identifier at the end
    clean_id = task_id
    if task_id.endswith('_Q1') or task_id.endswith('_Q2') or task_id.endswith('_Q3'):
        clean_id = task_id[:-3]
    
    # 1. Build scene image path
    # clean_id format: "3_generalized_planning/cross_embodiment/dual_arm/images/xxx"
    scene_image_path = os.path.join(dataset_base_dir, clean_id, "frame_00.png")
    
    # Check if frame_00.png exists, if not try other possible formats
    if not os.path.exists(scene_image_path):
        # Try cam_high_0.png (multi-view case)
        scene_image_path_alt = os.path.join(dataset_base_dir, clean_id, "cam_high_0.png")
        if os.path.exists(scene_image_path_alt):
            scene_image_path = scene_image_path_alt
        else:
            # Try directly as image file (color, shape, size)
            scene_image_path_alt2 = f"{os.path.join(dataset_base_dir, clean_id)}.png"
            if os.path.exists(scene_image_path_alt2):
                scene_image_path = scene_image_path_alt2
            else:
                print(f"Warning: scene image does not exist: {scene_image_path}")
    
    # Add scene image
    if os.path.exists(scene_image_path):
        image_paths.append(scene_image_path)
    else:
        print(f"Warning: cannot find scene image, task_id: {task_id}")
    
    # 2. Add DAG image
    if dag_image_path and os.path.exists(dag_image_path):
        image_paths.append(dag_image_path)
    else:
        print(f"Warning: DAG image does not exist or not provided, task_id: {task_id}")
    
    return image_paths

# Step 3: Extract Key Action (from extract_key_action.py) 
def statistics_final_score(dag_data=None):
    """Extract key actions and calculate final scores"""
    print("Starting statistics_final_score step...")
    
    if dag_data is not None:
        # Use data in memory
        data_sources = dag_data
    else:
        # Fall back to file reading
        file_list = glob(f"{eval_data_dir}/result/generate_dag_*")
        data_sources = {}
        for file_path in file_list:
            with open(file_path, 'r') as file:
                data_sources[file_path.split('generate_dag_')[-1]] = json.load(file)
    
    final_scores = {}
    
    for file_key, data in data_sources.items():
        
        final_score = 0
        final_length = 0
        
        for i in range(len(data)):
            # Try using new scoring fields, if not exist, fall back to old scoring fields
            try:
                # New scoring logic: two dimensions, each 0-10 points, converted to 100 points
                score = (data[i]["final_score"]['node_correctness']['result'] + \
                        data[i]["final_score"]['task_completion_degree']['result']) / 20.0 * 100.0
                final_score += score
            except KeyError:
                try:
                    # Mid-term scoring logic: three dimensions, converted to 100 points
                    score = (data[i]["final_score"]['node_correctness']['result'] + \
                            data[i]["final_score"]['task_sequence_reasonableness']['result'] + \
                            data[i]["final_score"]['task_completion_degree']['result']) / 15.0 * 100.0
                    final_score += score
                except KeyError:
                    # Fall back to oldest scoring logic: converted to 100 points
                    score = (data[i]["final_score"]['skill usage accuracy']['result'] * 5 + \
                            data[i]["final_score"]['operation object rationality']['result'] * 5 + \
                            data[i]["final_score"]['task sequence rationality']['result'] * 5 + \
                            data[i]["final_score"]['task completion']['result'] * 5) / 20.0 * 100.0
                    final_score += score
            final_length += 1

        if final_length > 0:
            mean_score = round(final_score / final_length, 2)
            print(f"{file_key}: score_mean : {mean_score:.2f}/100; length: {final_length}")
            with open(f"{eval_data_dir}/result.txt", 'a') as f_score:
                f_score.write(f"final_score_{file_key}: score_mean : {mean_score:.2f}/100; length: {final_length}\n")
            final_scores[file_key] = {"mean_score": mean_score, "length": final_length}
        else:
            print(f"{file_key}: no data to count")
            with open(f"{eval_data_dir}/result.txt", 'a') as f_score:
                f_score.write(f"final_score_{file_key}: no data to count\n")
            final_scores[file_key] = {"mean_score": 0, "length": 0}
    
    # Save final scores
    save_intermediate_result(final_scores, "final_scores")
    
    return final_scores

# Step 3: Calculate Final Scores for Q1, Q2, Q3 separately
def statistics_final_score_by_type(q1_dag_data=None, q2_results=None, q3_results=None):
    print("Starting statistics_final_score_by_type step...")
    
    final_scores = {
        'q1': {},
        'q2': {},
        'q3': {},
        'overall': {}
    }
    
    # Calculate Q1 scores (DAG evaluation results)
    if q1_dag_data is not None:
        print("Calculate Q1 scores...")
        for file_key, data in q1_dag_data.items():
            if not data:
                continue
                
            total_score = 0
            count = 0
            individual_scores = []
            
            for record in data:
                try:
                    score_dict = record["final_score"]
                    
                    # Extract new two scores (each 0-10 points)
                    node_correctness = score_dict.get('node_correctness', {}).get('result', 0)
                    completion_degree = score_dict.get('task_completion_degree', {}).get('result', 0)
                    
                    # Calculate total score (new scoring logic, converted to 100 points)
                    task_total = (node_correctness + completion_degree) / 20.0 * 100.0
                    task_total = round(task_total, 2)
                    
                    total_score += task_total
                    count += 1
                    
                    # Extract planning issue information (if exists)
                    planning_issues = record.get("planning_issues", {})
                    issue_types = planning_issues.get("issue_types", [])
                    
                    individual_scores.append({
                        'id': record["id"],
                        'node_correctness': node_correctness,
                        'completion_degree': completion_degree,
                        'total': task_total,
                        'planning_issue_types': issue_types,
                        'planning_issue_analysis': planning_issues.get("detailed_analysis", "")
                    })
                    
                except Exception as e:
                    print(f"Q1 calculation failed for record: {e}")
                    continue
            
            if count > 0:
                mean_score = round(total_score / count, 2)
                
                # Count issue types
                issue_type_counts = {}
                for score in individual_scores:
                    for issue_type in score.get('planning_issue_types', []):
                        issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1
                
                print(f"Q1 {file_key}: average score: {mean_score:.2f}/100, number of tasks: {count}")
                if issue_type_counts:
                    print(f"  Common issue types: {dict(sorted(issue_type_counts.items(), key=lambda x: x[1], reverse=True))}")
                
                final_scores['q1'][file_key] = {
                    "mean_score": mean_score,
                    "count": count,
                    "individual_scores": individual_scores,
                    "issue_type_statistics": issue_type_counts
                }
                
                # Save to file
                with open(f"{eval_data_dir}/result.txt", 'a') as f_score:
                    f_score.write(f"Q1_final_score_{file_key}: score_mean: {mean_score:.2f}/100; length: {count}\n")
                    if issue_type_counts:
                        f_score.write(f"Q1_issues_{file_key}: {dict(sorted(issue_type_counts.items(), key=lambda x: x[1], reverse=True))}\n")
            else:
                print(f"Q1 {file_key}: no valid data")
                final_scores['q1'][file_key] = {"mean_score": 0, "count": 0, "individual_scores": []}
    
    # Calculate Q2 scores
    if q2_results is not None:
        print("Calculate Q2 scores...")
        for file_key, data in q2_results.items():
            if not data:
                continue
                
            total_score = 0
            count = 0
            individual_scores = []
            
            for record in data:
                try:
                    score = record.get("score", 0)
                    total_score += score
                    count += 1
                    
                    individual_scores.append({
                        'id': record["task_id"],
                        'score': score,
                        'details': record.get('details', {})
                    })
                    
                except Exception as e:
                    print(f"Q2 calculation failed for record: {e}")
                    continue
            
            if count > 0:
                mean_score = round(total_score / count, 2)
                print(f"Q2 {file_key}: average score: {mean_score:.2f}/100, number of tasks: {count}")
                
                final_scores['q2'][file_key] = {
                    "mean_score": mean_score,
                    "count": count,
                    "individual_scores": individual_scores
                }
                
                # Save to file
                with open(f"{eval_data_dir}/result.txt", 'a') as f_score:
                    f_score.write(f"Q2_final_score_{file_key}: score_mean: {mean_score:.2f}/100; length: {count}\n")
            else:
                print(f"Q2 {file_key}: no valid data")
                final_scores['q2'][file_key] = {"mean_score": 0, "count": 0, "individual_scores": []}
    
    # Calculate Q3 scores
    if q3_results is not None:
        print("Calculate Q3 scores...")
        for file_key, data in q3_results.items():
            if not data:
                continue
                
            total_score = 0
            count = 0
            individual_scores = []
            
            for record in data:
                try:
                    score = record.get("score", 0)
                    total_score += score
                    count += 1
                    
                    individual_scores.append({
                        'id': record["task_id"],
                        'score': score,
                        'details': record.get('details', {})
                    })
                    
                except Exception as e:
                    print(f"Q3 calculation failed for record: {e}")
                    continue
            
            if count > 0:
                mean_score = round(total_score / count, 2)
                print(f"Q3 {file_key}: average score: {mean_score:.2f}/100, number of tasks: {count}")
                
                final_scores['q3'][file_key] = {
                    "mean_score": mean_score,
                    "count": count,
                    "individual_scores": individual_scores
                }
                
                # Save to file
                with open(f"{eval_data_dir}/result.txt", 'a') as f_score:
                    f_score.write(f"Q3_final_score_{file_key}: score_mean: {mean_score:.2f}/100; length: {count}\n")
            else:
                print(f"Q3 {file_key}: no valid data")
                final_scores['q3'][file_key] = {"mean_score": 0, "count": 0, "individual_scores": []}
    
    # Calculate overall scores
    print("\nCalculate overall scores...")
    all_file_keys = set()
    if q1_dag_data:
        all_file_keys.update(q1_dag_data.keys())
    if q2_results:
        all_file_keys.update(q2_results.keys())
    if q3_results:
        all_file_keys.update(q3_results.keys())
    
    for file_key in all_file_keys:
        q1_score = final_scores['q1'].get(file_key, {}).get('mean_score', 0)
        q1_count = final_scores['q1'].get(file_key, {}).get('count', 0)
        q2_score = final_scores['q2'].get(file_key, {}).get('mean_score', 0)
        q2_count = final_scores['q2'].get(file_key, {}).get('count', 0)
        q3_score = final_scores['q3'].get(file_key, {}).get('mean_score', 0)
        q3_count = final_scores['q3'].get(file_key, {}).get('count', 0)
        
        total_count = q1_count + q2_count + q3_count
        if total_count > 0:
            # Calculate weighted average score
            overall_score = (q1_score * q1_count + q2_score * q2_count + q3_score * q3_count) / total_count
            overall_score = round(overall_score, 2)
            
            final_scores['overall'][file_key] = {
                "mean_score": overall_score,
                "total_count": total_count,
                "q1_count": q1_count,
                "q2_count": q2_count, 
                "q3_count": q3_count,
                "q1_score": q1_score,
                "q2_score": q2_score,
                "q3_score": q3_score
            }
            
            print(f"Overall {file_key}: total score: {overall_score:.2f}/100, total number of tasks: {total_count} (Q1:{q1_count}, Q2:{q2_count}, Q3:{q3_count})")
            
            # Save to file
            with open(f"{eval_data_dir}/result.txt", 'a') as f_score:
                f_score.write(f"Overall_final_score_{file_key}: score_mean: {overall_score:.2f}/100; total_length: {total_count} (Q1:{q1_count}, Q2:{q2_count}, Q3:{q3_count})\n")
    
    # Save final scores details
    save_intermediate_result(final_scores, "final_scores_by_type")
    
    return final_scores

async def process_directory(dir_path, use_cache=True, skip_steps=None, dataset_base_dir=None, base_url=base_url, api_key=api_key):
    """Process a single directory
    
    Args:
        dir_path: Path to the directory to process
        use_cache: Whether to use cached intermediate results
        skip_steps: List of steps to skip
        dataset_base_dir: Dataset base directory path
        base_url: Base URL for the API
        api_key: API key
    """
    print(f"\nProcessing directory: {dir_path}")
    print(f"Dataset base directory: {dataset_base_dir}")
    
    # Clear DAG cache to avoid data mixing between different directories
    clear_dag_cache()
    
    # Set the intermediate and result directories for this directory
    intermediate_dir, result_dir = setup_directories(dir_path)
    
    # Set global variables because other functions are still using these variables
    global eval_data_dir, intermediate_results_dir
    eval_data_dir = dir_path
    intermediate_results_dir = intermediate_dir
    
    if 'classify' not in skip_steps:
        classified_data = await classify_and_extract_planning_data(use_cache=use_cache)
    else:
        print("Skip classification step, load cached data")
        classified_data = load_intermediate_result("classified_planning_data")
    
    if not classified_data:
        print("No classified data to process")
        return None
        
    # Process Q1 tasks
    q1_extracted_data = None
    q1_dag_data = None
    
    if 'q1_extract' not in skip_steps:
        print("=== Process Q1 tasks - extract plan information ===")
        q1_extracted_data = await extract_plan_information_q1(classified_data, use_cache=use_cache, base_url=base_url, api_key=api_key)
    
    if 'q1_dag' not in skip_steps and q1_extracted_data:
        print("=== Process Q1 tasks - DAG generation ===")
        q1_dag_data = await dag_llm_evaluation_q1(q1_extracted_data, use_cache=use_cache, dataset_base_dir=dataset_base_dir, base_url=base_url, api_key=api_key)
    
    # Process Q2 tasks
    q2_results = None
    if 'q2' not in skip_steps:
        print("=== Process Q2 tasks ===")
        q2_results = await process_q2_tasks(classified_data, use_cache=use_cache, base_url=base_url, api_key=api_key)
    
    # Process Q3 tasks
    q3_results = None
    if 'q3' not in skip_steps:
        print("=== Process Q3 tasks ===")
        q3_results = await process_q3_tasks(classified_data, use_cache=use_cache, base_url=base_url, api_key=api_key)
    
    # Calculate final scores
    if 'final_score' not in skip_steps:
        print("=== Calculate final scores ===")
        final_scores = statistics_final_score_by_type(
            q1_dag_data=q1_dag_data,
            q2_results=q2_results,
            q3_results=q3_results
        )
    else:
        final_scores = None
    
    return {
        'classified_data': classified_data,
        'q1_extracted_data': q1_extracted_data,
        'q1_dag_data': q1_dag_data,
        'q2_results': q2_results,
        'q3_results': q3_results,
        'final_scores': final_scores
    }

async def main(use_cache=True, skip_steps=None, input_dir=None, version=None, dataset_base_dir=None, base_url=base_url, api_key=api_key):
    """
    Main program entry - supports Q1, Q2, Q3 separately
    
    Args:
    use_cache (bool): Whether to use cached intermediate results
    skip_steps (list): List of steps to skip
    input_dir (str): Input data directory path
    version (str): Version subdirectory name
    dataset_base_dir (str): Dataset base directory path, used to read DAG information
    base_url: Base URL for the API
    api_key: API key
    """
    print("Starting enhanced all-in-one pipeline execution...")
    
    if not input_dir:
        input_dir = os.environ.get('EVAL_DATA', "")
    
    print(f"Input directory: {input_dir}")
    print(f"Dataset base directory: {dataset_base_dir}")
    print(f"Use cache: {use_cache}")
    if version:
        print(f"Version subdirectory: {version}")
    
    if skip_steps is None:
        skip_steps = []
    
    # Check the content of the input directory
    subdirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    all_results = {}
    
    if subdirs:
        print(f"Found subdirectories: {subdirs}")
        for subdir in subdirs:
            subdir_path = os.path.join(input_dir, subdir)
            print(f"\n{'='*60}\nProcessing subdirectory: {subdir_path}")
            result = await process_directory(subdir_path, use_cache, skip_steps, dataset_base_dir, base_url, api_key)
            if result:
                all_results[subdir] = result
                # After processing, move the outputs to the version subdirectory
                move_outputs_to_version(subdir_path, version)
    elif json_files:
        print(f"Found JSON files: {json_files}")
        # Directly process the current directory
        result = await process_directory(input_dir, use_cache, skip_steps, dataset_base_dir, base_url, api_key)
        if result:
            all_results['root'] = result
            # After processing, move the outputs to the version subdirectory
            move_outputs_to_version(input_dir, version)
    else:
        print("No files or folders to process in the input directory")
        return None
        
    return all_results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate planning results')
    
    parser.add_argument('--no-cache', action='store_false', dest='use_cache',
                      help='Do not use cached intermediate results')
    parser.add_argument('--input-dir', type=str, default=None,
                      help='Input folder path, if not specified, use EVAL_DATA environment variable')
    parser.add_argument('--skip-steps', nargs='+', choices=['classify', 'q1_extract', 'q1_dag', 'q2', 'q3', 'final_score'],
                      default=[], help='Type of steps to skip')
    parser.add_argument('--version', type=str, default=None,
                      help='Version subdirectory name, e.g. v1, 20240930, etc.')
    parser.add_argument('--dataset-base-dir', type=str, default='Your own path to the RoboBench-hf dataset',
                      help='Dataset base directory path, default is your own path to the RoboBench-hf dataset')
    parser.add_argument('--base-url', type=str, default='Your own base URL for the API',
                      help='Base URL for the API')
    parser.add_argument('--api-key', type=str, default='Your own OpenAI API key',
                      help='API key')
    args = parser.parse_args()
    
    print("Starting parameters:")
    print(f"  Use cache: {args.use_cache}")
    print(f"  Skip steps: {args.skip_steps}")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Dataset base directory: {args.dataset_base_dir}")
    print(f"  Version subdirectory: {args.version}")
    print(f"  Base URL: {args.base_url}")
    print(f"  API key: {args.api_key}")

    result = asyncio.run(main(
        use_cache=args.use_cache,
        skip_steps=args.skip_steps,
        input_dir=args.input_dir,
        version=args.version,
        dataset_base_dir=args.dataset_base_dir,
        base_url=args.base_url,
        api_key=args.api_key
    ))
    print("\nProgram executed successfully!")
