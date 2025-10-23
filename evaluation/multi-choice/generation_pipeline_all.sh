# Run model inference
models=("gpt-4o")

# Base directory for data
BASE_DIR="Your own path to the RoboBench-hf dataset"

OPENAI_API_KEY="Your own OpenAI API key"

BASE_URL="Your own base URL for the API"

# Create results directory if it doesn't exist
RESULTS_DIR="Your own path to the multi-choice results directory"
mkdir -p "$RESULTS_DIR"

# Create log file with timestamp
LOG_FILE="$RESULTS_DIR/generation_pipeline_$(date +%Y%m%d_%H%M%S).log"
echo "Saved log file to: $LOG_FILE"

# Redirect all output to both terminal and log file
exec > >(tee -a "$LOG_FILE") 2>&1

# Target directories to search
TARGET_DIRS=("2_perception_reasoning" "4_affordance_reasoning" "5_error_analysis")

for model in "${models[@]}"; do
    # Find all questions.json files in the target directories
    for target_dir in "${TARGET_DIRS[@]}"; do
        while IFS= read -r -d '' questions_file; do
            cd ../api_call_general2
            echo $model $questions_file
            filename=$(basename "$questions_file")
            output_filename="${filename%.*}.jsonl"

            # Extract relative path from directory for organizing results
            relative_path=$(realpath --relative-to="$BASE_DIR" "$questions_file")
            dir_path=$(dirname "$relative_path")

            python3 general_pipeline.py \
                --base-url="$BASE_URL" \
                --api-key="$OPENAI_API_KEY" \
                --questions_file="$questions_file" \
                --output_file="middle_file/${dir_path//\//_}_${output_filename}" \
                --result-dir="$RESULTS_DIR" \
                --system_prompt_file="$BASE_DIR/system_prompt.json" \
                --model="$model" \
                --image_key="image_urls" \
                --question_key="question"

            cd ../multi-choice
            python3 merge_all_results.py --model="$model" --file="$questions_file" --output-dir="$RESULTS_DIR"
            echo $(basename $(dirname $questions_file))
            python3 evaluate_responses.py --results_file="${RESULTS_DIR}/$(basename $(dirname $questions_file))/${model}_result_${filename}" --output="./results/$(basename $(dirname $questions_file))/evaluation_${model}_results_${filename}" \
            --openai-api-key="$OPENAI_API_KEY" --base-url="$BASE_URL"
            
        done < <(find "$BASE_DIR/$target_dir" -name "questions.json" -type f -print0)
    done
done

