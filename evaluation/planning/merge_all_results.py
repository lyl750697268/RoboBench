import json
import os
import argparse
def merge_results_for_question(output_dir: str, model: str = "gemini", file: str = ""):
    """Merge results for a specific question ID"""
    # Find model result file for this question
    model_file = None

    model_prefix = model
    for filename in os.listdir(output_dir):
        if filename.startswith(model_prefix) and os.path.basename(file) in filename:
            model_file = filename
            break
    
    if not model_file:
        print(f"No {model} result file for {os.path.basename(file)}")
        return
        
    # Load model data
    with open(os.path.join(output_dir, model_file), 'r') as f:
        model_data = json.load(f)
    
    # Load processed data
    processed_path = file
    
    if not os.path.exists(processed_path):
        print(f"No processed file found: {os.path.basename(file)}")
        return
        
    with open(processed_path, 'r') as f:
        processed_data = json.load(f)
        
    # Create mapping of id to processed data
    processed_map = {item["unique_id"]: item for item in processed_data}
    
    # Merge data
    for item in model_data:
        if item is None:
            continue
        item_id = item.get("id")
        if item_id in processed_map:
            # Add fields from processed data
            processed_item = processed_map[item_id]
            item["question"] = processed_item["question"]
            item["gt_answer"] = processed_item["gt_answer"]
            # item["image_urls"] = processed_item["image_urls"]
            # if "previous_step" in processed_item:
            #     item["previous_step"] = processed_item["previous_step"]
    
    # Save merged results
    output_filename = model_file
    with open(os.path.join(output_dir, output_filename), 'w') as f:
        json.dump(model_data, f, indent=2)
    print(f"Merged results saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge results for all questions")
    parser.add_argument("--model", type=str, default="gemini", help="Model to use for merging")
    parser.add_argument("--file", type=str, default="", help="File name to use for merging")
    args = parser.parse_args()
    output_dir = "../../results/" + args.file.split('/')[-2]
    merge_results_for_question(output_dir, args.model, args.file)