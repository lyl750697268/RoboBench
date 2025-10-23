# -*- coding: utf-8 -*-
import asyncio
import aiofiles
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import json, os

SET_SERVER_VERSION = 0

base_url="https://api.openai.com/v1"
api_key="Your own OpenAI API key"

MAX_CONCURRENT = 50

model = "gpt-4o"

def retry_error_callback(retry_state):
    exception = retry_state.outcome.exception()
    print(f"Retry attempt {retry_state.attempt_number} failed: {type(exception).__name__} - {str(exception)}")
    return None

async def save_temp_results(results, save_path, current_count):
    temp_file = f"{save_path}_temp_result.json"
    last_file = f"{save_path}_temp_result_last.json"
    
    if os.path.exists(temp_file):
        try:
            os.replace(temp_file, last_file)
        except Exception as e:
            print(f"Failed to rename old temp file: {e}")
    
    async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(results, ensure_ascii=False, indent=2))

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=1, max=10), retry_error_callback=retry_error_callback)
async def get_chat_completion(messages: dict, request_id, semaphore, api_model=model, base_url=base_url, api_key=api_key, retry_count=0) -> str:
    response = None
    resp = {'id': request_id}
    try:
        async with semaphore:
            response = await AsyncOpenAI(
                base_url=base_url, api_key=api_key, timeout=360
            ).chat.completions.create(
                model=api_model, messages=messages, timeout=360
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


async def request_model(prompts, request_ids, api_model=model, base_url=base_url, api_key=api_key, save_path="results"):
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    results = [None] * len(prompts)
    completed_count = 0
    failed_count = 0
    
    TASK_TIMEOUT = 960

    async def wrapped_get_chat_completion(prompt, request_id, index, api_model=model, base_url=base_url, api_key=api_key):
        nonlocal completed_count, failed_count
        try:
            result = await asyncio.wait_for(
                get_chat_completion(prompt, request_id, semaphore, api_model, base_url, api_key),
                timeout=TASK_TIMEOUT
            )
            completed_count += 1
            
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

    tasks = [wrapped_get_chat_completion(prompt, request_ids[i], i, api_model, base_url, api_key) for i, prompt in enumerate(prompts)]
    
    print(f"Starting to process {len(tasks)} tasks, maximum concurrent requests: {MAX_CONCURRENT}")
    
    completed_tasks = 0
    for future in asyncio.as_completed(tasks):
        try:
            index, result = await future
            results[index] = result
            completed_tasks += 1

            if completed_tasks % 50 == 0 or completed_tasks == len(tasks):
                print(f"Progress: {completed_tasks}/{len(tasks)} ({completed_tasks/len(tasks)*100:.1f}%) - Success: {completed_count}, Failed: {failed_count}")
                
        except Exception as e:
            print(f"Unexpected error in task completion: {e}")
            continue

    print(f"All tasks completed! Success: {completed_count}, Failed: {failed_count}")
    await save_temp_results(results, save_path, "final")
    return results
