import json, os

def prompt_no_template(prompt_file, output_file, instruction_key=None, prompt_key='question', image_key=None, system_prompt_file=None):
    with open(prompt_file, 'r') as fi, open(output_file, 'w') as fo:
        if prompt_file.endswith('.json'):
            original_prompts = json.load(fi)
        else:
            original_prompts = [json.loads(line) for line in fi]
        original_prompts = [p for p in original_prompts if isinstance(p, dict)]
            
        for idx, prompt in enumerate(original_prompts):
            _prompt = {}
            _id = idx
            if 'unique_id' in prompt.keys():
                _id = prompt['unique_id']
            elif 'id' in prompt.keys():
                _id = prompt['id']
            _prompt['request_id'] = _id
            question_type = prompt_file.split('/')
            if system_prompt_file is not None:
                with open(system_prompt_file, 'r') as f:
                    system_prompt = json.load(f)
                
                if '3_generalized_planning' in question_type or '1_instruction_comprehension' in question_type or 'temporal_event_order' in question_type:
                    if 'navigation' in question_type:
                        _prompt['prompt'] = system_prompt['navigation_skill_list'] + prompt[prompt_key]
                    else:
                        if 'Q1' in _id:
                            _prompt['prompt'] = system_prompt['skill_list_Q1'] + prompt[prompt_key]
                        elif 'Q2' in _id:
                            _prompt['prompt'] = system_prompt['skill_list_Q2'] + prompt[prompt_key]
                        elif 'Q3' in _id:
                            _prompt['prompt'] = system_prompt['skill_list_Q3'] + prompt[prompt_key]
                        else:
                            _prompt['prompt'] = system_prompt['skill_list_Q1'] + prompt[prompt_key]
                    if 'robotic_type' in prompt:
                        if prompt['robotic_type'] == "human":
                            robotic_type = "single-arm"
                        else:
                            robotic_type = prompt['robotic_type']
                        _prompt['prompt'] += f"You are a {robotic_type} robot." 
                    elif 'temporal_event_order' in question_type:
                        _prompt['prompt'] += f"You are a single-arm robot." 
                    if 'multi_view' in question_type:
                        views = [img.split('.')[0] for img in prompt["image_urls"]]
                        _prompt['prompt'] += f"The multiple images you have received now represent camera images from different views at the same time. The names of these views are {views} in order."
                elif 'static_attribute' in question_type:
                    _prompt['prompt'] = system_prompt[_id.split('/')[-2]] + prompt[prompt_key]
                elif 'robot_type' in question_type:
                    _prompt['prompt'] = system_prompt['robot_type'] + prompt[prompt_key]
                elif 'robot_view' in question_type:
                    _prompt['prompt'] = system_prompt['robot_view'] + prompt[prompt_key]
                elif 'spatial_relation' in question_type:
                    _prompt['prompt'] = system_prompt['spatial_relation'] + prompt[prompt_key]
                elif 'spatial_temporal_causality' in question_type:
                    _prompt['prompt'] = system_prompt['spatial_temporal_causality'] + prompt[prompt_key]
                elif 'static_affordance' in question_type or 'dynamic_affordance' in question_type or 'navigation_visual_prompt' in question_type:
                    _prompt['prompt'] = system_prompt['afforadance_point'] + prompt[prompt_key]
                elif 'color' in question_type:
                    _prompt['prompt'] = system_prompt['color'] + prompt[prompt_key]
                elif 'number' in question_type:
                    _prompt['prompt'] = system_prompt['number'] + prompt[prompt_key]
                elif 'shape' in question_type:
                    _prompt['prompt'] = system_prompt['shape'] + prompt[prompt_key]
                elif 'size' in question_type:
                    _prompt['prompt'] = system_prompt['size'] + prompt[prompt_key]
                elif 'high_level_planning_error' in question_type:
                    _prompt['prompt'] = system_prompt['high_level_planning_error'] + prompt[prompt_key]
                elif 'low_level_execution_error' in question_type:
                    _prompt['prompt'] = system_prompt['low_level_execution_error'] + prompt[prompt_key]
                else:
                    _prompt['prompt'] = prompt[prompt_key]
                
                # post process
                choices = prompt.get("options", [])
                base = ord("A")
                for i, choice in enumerate(choices):
                    _prompt['prompt'] += "\n" + chr(base + i) + ". " + choice
                if choices != []:
                    _prompt['prompt'] += system_prompt["MCQ_post_prompt"]

            if instruction_key is not None:
                _prompt['instruction'] = prompt[instruction_key]

            if image_key is not None:
                original_images = prompt[image_key]
                if isinstance(_id, str) and 'static_attribute' in _id:
                    mapped_images = [f"../../RoboBench-hf/{_id}/{img}" for img in original_images]
                else:
                    if prompt["input_type"] == "image":
                        mapped_images = [f"../../RoboBench-hf/{'/'.join(_id.split('/')[:-1])}/{img}" for img in original_images]
                    else:
                        mapped_images = [f"../../RoboBench-hf/{_id.split('_Q')[0]}/{img}" for img in original_images]
                _prompt['image_urls'] = mapped_images
            fo.write(json.dumps(_prompt, ensure_ascii=False) + '\n')

def prompt_with_template(prompt_file, output_file, prompt_template, image_key=None):
    with open(prompt_file, 'r') as fi, open(output_file, 'w') as fo:
        if prompt_file.endswith('.json'):
            original_prompts = json.load(fi)
        else:
            original_prompts = [json.loads(line) for line in fi]
        original_prompts = [p for p in original_prompts if isinstance(p, dict)]
        for idx, prompt in enumerate(original_prompts):
            _prompt = {}
            _id = idx
            if 'id' in prompt.keys():
                _id = prompt['id']
            elif 'request_id' in prompt.keys():
                _id = prompt['request_id']
            _prompt['request_id'] = _id
            try:
                _text_prompt = prompt_template.format(**prompt)
            except:
                raise ValueError("Error in formatting prompt with template: "+ prompt_template)
            _prompt['prompt'] = _text_prompt
            if image_key is not None:
                _prompt['image_urls'] = prompt[image_key]
            fo.write(json.dumps(_prompt, ensure_ascii=False) + '\n')

def format_messages(messages, mode='base64'):
    prompt_messages = []
    request_ids = []
    if not isinstance(messages, list):
        messages = [messages]
    for message in messages:
        prompt_message = []
        assert 'request_id' in message.keys()
        assert 'prompt' in message.keys()
        
        # setting up system prompt
        if 'instruction' in message.keys():
            prompt_message.append(
                {"role": "system", "content": message["instruction"]},
            )

        # setting up user prompt
        user_content = []
        user_content.append(
            {"type": "text", "text": message["prompt"]},
        )
        if 'image_urls' in message.keys():
            if mode == 'base64':
                from cv_utils import reshape_frame_to_512, image_to_frame, encode_frame_cv
            for i, img in enumerate(message['image_urls']):
                user_content.append(
                    {
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_frame_cv(image_to_frame(img))}"
                        }
                    }
                )
        prompt_message.append(
            {"role": "user", "content": user_content}
        )
        prompt_messages.append(prompt_message)
        request_ids.append(message["request_id"])
    return prompt_messages, request_ids