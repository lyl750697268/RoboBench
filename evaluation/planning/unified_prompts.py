########################
# PREDEFINED_ACTIONS
########################
PREDEFINED_ACTIONS = """
#### **Functions for the actions of a gripper**
  -  `move_to(object, target_object)`: This function represents the movement of the gripper, with the first parameter representing the object held in the hand during movement and the second parameter representing the target object. If there is nothing in hand, the first parameter is 'none'. For example, move_to(none, towel) means that the gripper is moving towards a towel with nothing in hand. And move_to(panda_toy, bowl) means that the gripper with a panda toy is moving towards a bowl.
  -  `hold(object)`: This function represents the static state of the gripper with an object. Note that if there is nothing in gripper, this function is not applicable. For example, hold(cup) means that the gripper is holding a cup and keeping static.

#### **Functions for grabbing and releasing**
  - `pick_up(object)`: This function represents that the gripper picks up an object. Note that the object must be graspable. For example, pick_up(apple) means that the gripper picks up an apple.
  - `grasp(object)`: This function represents that the gripper touches and lightly grabs the object. Note that objects can be either pick-upable or non-pick-upable. The difference between this function and pick_up is that grasp means to hold lightly, while pick_up usually means to lift up after contact. For example, grasp(door_handle) means that the gripper grabs the door handle (not lift up).
  - `place(object, target_object)`: This function represents that the gripper place the object at the location of the target_object. Note that target_object can be a specific object or a position relative to a reference object. For example, place(apple, table) means that placing an apple on the table. And place(apple, right_of_banana) means that placing an apple on the right of a banana.

#### **Functions for using a tool to operate objects**
  - `scoop(tool, contents, container)`: This function represents that the gripper holding the tool is scooping something in a container with the tool. If "contents" is uncertain, use "unknown" as the second parameter. For example, scoop(spoon, water, bowl) means that the gripper holds the spoon and uses it to scoop water from the bowl.
  - `pour(container, contents, target_container)`: This function represents that the gripper holding the container is pouring something into the target_container. If "contents" is uncertain, use "unknown" as the second parameter. For example, pour(bowl, water, pot) means that the gripper holds a bowl with water and pours water into a pot.
  - `wipe(tool, object, target_object)`: This function represents that the gripper is using a tool to wipe "object" on the "target_object". For example, wipe(towel, water, table) means that the gripper is wiping water on the table using a towel. If "object" is uncertain, use "unknown" instead.
  - `stir(tool, contents, target_container)`: This function represents that the gripper is using a tool to stir "contents" in the "target_container". For example, stir(spoon, soup, pot) means that using a spoon to stir soup in the pot. If "contents" is uncertain, use "unknown" instead.
  - `draw(tool, character, target_object)`: This function represents that the gripper is drawing a character using a tool on the target_object. For example, draw(marker, 'A', whiteboard) means that drawing an 'A' with a marker on the whiteboard.
  - `cut(tool, object, target_object)`: This function represents that the gripper is cutting object with a tool on the target_object. For example, cut(knife, tomato, chopping_board) means that the gripper is cutting a tomato with a knife on the chopping_board.

#### **Functions for interacting with objects directly**
  -  `fold(object, target_position)`: This function represents that the gripper is folding object towards the target_position. For example, fold(left_side_of_towel, right_side_of_towel) means that the gripper holds the left side of the towel and folds it to the right side.
  -  `unfold(object, target_position)`: This function represents that the gripper is unfolding object towards the target_position. For example, unfold(left_side_of_towel, right_side_of_towel) means that the gripper holds the left side of the towel and unfolds it to the right side.
  -  `turn(object, direction, state_of_target_object)`: This function represents that the gripper rotates the object in a certain direction to the target position or state. The 'direction' can be chosen in {{clockwise, anticlockwise, up, down, forward, backward, left, right}}. If it is an articulated object (it originally has axis rotation, such as a faucet), use the following {{up, down, forward, backward, left, right}}. If it is originally a rigid body (it originally has no axis rotation, such as a bottle adjusting its direction in the air), use clockwise and counterclockwise. For example, turn(faucet, clockwise, middle_of_sink) means that turn faucet right until it faces the middle of sink.
  -  `press(tool, object)`: This function represents that the gripper press object using a tool. If there is no tool, use 'none' instead. For example, press(none, red_button) means that pressing a red button using the gripper directly.
  -  `push(object, target_location)`: This function represents that the gripper pushes the object to the target_location. For example, push(chair, under_of_table) means that pushing a chair under to a table.
  -  `pull(object, target_location)`: This function represents that the gripper pulls the object to the target_location. For example, pull(towel, right_side_of_table) means that pulling a towel to the right side of a table.
  -  `insert(object, target_object)`: This function represents that the gripper inserts the object into the target object. For example, insert(plug, socket) means inserting a plug into the socket.
  -  `pullout(object, target_object)`: This function represents that the gripper pullouts the object from the target object. For example, pullout(plug, socket) means pullout a plug from the socket.

#### **Function for only dual-arm tasks**
  -  `transfer(left/right, right/left, object)`: This function represents the gripper transfers an object from one hand to another hand. For example, transfer(left, right, bottle) means that transfering bottle from left hand to right hand.

#### **Functions for only mobile-manipulation tasks**
  -  `observation(object)`: This function represents that the target object is not in the field of view and needs to be found. For example, observation(chair) means that finding the chair.
  -  `mobile(target_object)`: This function represents that the target object is in the field of view but too far from the robot to operate, so the robot needs to move the chassis to approach the target position. For example, mobile(table) means moving the chassis to approach the table.

#### **No Operation**
  - `no_ops`: Stay still or keep the current state.
"""

########################
# PROMPT_TEMPLATE_EXTRACT_PLAN_INFO
########################
PROMPT_TEMPLATE_EXTRACT_PLAN_INFO = """
### **Prompt: Extracting Structured Action Plans from Robotic Task Descriptions**

**Task:**
You are given an input dataset containing a robotic manipulation task goal, a previously executed step, and a response describing the remaining steps. Your task is to extract structured action plans in a specific function format.

---

### **Instructions:**
1. **Extract Key Information:**
   - Identify the **task goal** from the `prompt` field and assign it to the `"task_summary"` field.
   - Extract action functions from the `previous_step` and `response` fields to construct a sequence of necessary steps in `"plan_step"`.

2. **Strict Action Function Format:**
   - Use only the predefined action functions listed below. **Do not modify function names or introduce new ones.**
   - Ensure that all extracted function names match exactly with the ones provided.
   - Arguments (`object`, `target_object`, `carry_object`, `direction`) should be **generalized** based on input information but remain **faithful** to the task.

3. **Maintain Execution Order:**
   - The `"plan_step"` list should strictly follow the order in which the robot should execute them.

4. **Determine Action Format Based on Input:**
   - **Single-arm tasks**: If the `response` field contains actions without `left:` or `right:` prefixes, extract actions in single-arm format (e.g., `move_to(object, target)`).
   - **Dual-arm tasks**: If the `response` field contains actions with `left:` or `right:` prefixes, extract actions in dual-arm format (e.g., `left:move_to(object, target), right:no_ops`).
   - **Automatic detection**: Analyze the input content to determine whether it's a single-arm or dual-arm task and format the output accordingly.

5. **Strictly No Assumptions:**
   - Only extract actions explicitly present in the input.
   - Do **not** add missing steps based on assumptions.
   - **Do not reinterpret or correct possibly incorrect actions or arguments** — preserve the input as given.

6. **Reasoning Explanation:**
   - Provide a `"reason"` field explaining how the `"task_summary"` and `"plan_step"` were derived, including how you determined the format (single-arm vs dual-arm).

---

### **Predefined Action Functions**
""" + PREDEFINED_ACTIONS + """

---

### **Output Format (JSON)**
Your response **must be strictly formatted** as a single-line JSON object:

**For Single-Arm Tasks:**
```json
{{
  "task_summary": "<task goal>",
  "plan_step": ["<action_function_1>", "<action_function_2>", ...],
  "reason": "<your reasoning>"
}}
```

**For Dual-Arm Tasks:**
```json
{{
  "task_summary": "<task goal>",
  "plan_step": [
      "<action_function_1>",
      "left:<action_function_2>, right:<action_function_3>",
      ...
  ],
  "reason": "<your reasoning>"
}}
```

### **Examples**

#### **Example 1: Single-Arm Task**
**Input:**
```json
{{
  "prompt": "With <placing a roll of toilet paper onto a holder> as the goal and some steps completed, what are the remaining things to do?",
  "previous_step": "1-move_to(toilet_paper)",
  "response": "From the sequence of images, it appears that a robotic system is in the process of placing the roll of toilet paper onto its holder. Here are the remaining things to do:\\n\\n1. Align the Toilet Paper Roll with the Holder's Rod.  \\n2. Insert the Rod Through the Roll.  \\n3. Attach the Rod and Roll to the Holder Mechanism.  \\n4. Finalize the Setup."
}}
```

**Expected Output:**
```json
{{
  "task_summary": "<placing a roll of toilet paper onto a holder>",
  "plan_step": ["move_to(toilet_paper)", "place(toilet_paper, holder)"],
  "reason": "Detected single-arm task based on response format without left/right prefixes. The first step 'move_to(toilet_paper)' is obtained from the 'previous_step' field, and the second step 'place(toilet_paper, holder)' is derived from the 'prompt' and 'response' fields by summarizing the remaining steps."
}}
```

#### **Example 2: Dual-Arm Task**
**Input:**
```json
{{
  "prompt": "With <cooking shrimp in a pan and serving it in a bowl> as the goal and some steps completed, what are the remaining things to do?",  
  "previous_step": "1-observation(shrimp)",  
  "response": "To plan the remaining steps to achieve the goal of cooking shrimp in a pan and serving it in a bowl, we'll organize the steps required for the robot...\\n\\n1-left:move_to(none, shrimp), right:no_ops\\n2-left:pick_up(shrimp), right:no_ops\\n3-left:no_ops, right:move_to(none, pan)\\n4-left:no_ops, right:turn_on(stove)\\n5-left:no_ops, right:pour(oil, pan)\\n6-left:no_ops, right:move_to(shrimp, pan)\\n7-left:no_ops, right:scoop(shrimp, pan)\\n8-left:no_ops, right:move_to(pan, bowl)\\n9-left:no_ops, right:pour(shrimp, bowl)"
}}
```

**Expected Output:**
```json  
{{
  "task_summary": "<cooking shrimp in a pan and serving it in a bowl>",  
  "plan_step": [  
      "observation(shrimp)",  
      "left:move_to(none, shrimp), right:no_ops",  
      "left:pick_up(shrimp), right:no_ops",  
      "left:no_ops, right:move_to(none, pan)",  
      "left:no_ops, right:turn_on(stove)",  
      "left:no_ops, right:pour(oil, pan)",  
      "left:no_ops, right:move_to(shrimp, pan)",  
      "left:no_ops, right:scoop(shrimp, pan)",  
      "left:no_ops, right:move_to(pan, bowl)",  
      "left:no_ops, right:pour(shrimp, bowl)"  
  ],
  "reason": "Detected dual-arm task based on response format with left/right prefixes. The 'task_summary' field is extracted from the prompt. The 'plan_step' follows the order in 'previous_step' and 'response'. The 'observation' step is included as a standalone action, while all manipulation actions use the 'left:' and 'right:' format. No missing steps were assumed or inferred."
}}
```

The data I provide is as follows:
{data}
Please output your results as required.

### **Final Instructions**
- **Automatically detect task type**: Analyze the input to determine if it's single-arm or dual-arm and format accordingly.
- **Do not modify function names.**
- **Do not add missing steps beyond the input information.**
- **Ensure correct function selection based on context.**
- **Ensure the `"plan_step"` field is strictly ordered.**
- **For dual-arm tasks: All manipulation actions are formatted with `left:` and `right:` unless explicitly global actions.** If an arm does not perform any action in a step, use `no_ops`.
- **Standalone movement (`mobile`) and observation (`observation`) actions must NOT have `left:` or `right:` prefixes in dual-arm tasks.**
- **Output a single-line JSON object with no extra content.**
- **Do not reinterpret or correct possibly incorrect actions or arguments** — preserve the input as given.
"""

########################
# PROMPT_TEMPLATE_Q1_EVALUATION
########################
PROMPT_TEMPLATE_Q1_EVALUATION = """
You are now a judge of embodied multimodal task planning. Your task is to comprehensively evaluate and score the planning capabilities of the multimodal large model in embodied intelligence scenarios.

### Input information
You will receive the following:
* **An image representing the current scene state** - **CRITICAL**: This image provides the initial simulation conditions and visual constraints that must be considered during evaluation. The image shows object positions, obstacles, spatial relationships, and environmental constraints that affect plan feasibility;
* A Ground Truth (GT) Action List generated by Gemini by segmenting a static video;
* A Directed Acyclic Graph (DAG) manually annotated from the video, representing the sequential relationship between tasks (tasks can be ordered or parallel). The DAG is presented in both text and image form. The parallel relationship between subtasks is primarily based on the image, not the GT Action List. If the text is unclear, refer to the text for details;
* An Action List output by the large multimodal model to be evaluated. Note that the model inputs a static image of the current scene and the question, and outputs an Action List. We do not feed the model with a DAG; the DAG is used as a scoring reference.

### Input data format
{{
    'GT action list': [Ground Truth (GT) Action List obtained by Gemini through segmented description of static video],
    'GT dag': [DAG text description],
    'model plan action list': [An Action List output by the multimodal large model to be evaluated]
}}

### Automatic Mode Detection
**Automatically detect the evaluation mode based on the input data:**
- **Standard Mode**: If the action functions in the GT action list and model action list contain textual object names (e.g., `pick_up(apple)`, `move_to(table)`), use standard evaluation mode.
- **CSS Mode**: If the action functions contain numeric object IDs (e.g., `pick_up(3)`, `move_to(5)`), use CSS evaluation mode where objects must be referred to by their serial numbers as shown in the image.
- **Detection Logic**: Analyze the action functions in both GT action list and model action list to determine the appropriate mode and evaluation criteria.

### Predefined Action Functions
""" + PREDEFINED_ACTIONS + """

### Evaluation criteria

You are now acting as a visual-language world simulator. Based on the first image, GT action list, and manually annotated DAG, you need to build an understanding of the entire scene and task, then evaluate the model's planning using the following two dimensions (each scored 0-10 points, 10 points is the best and 0 points is the worst):

1. **Node correctness (0-10 points)**:
   - Treat each skill-object-parameter combination as a node
   - Count the proportion of completely correct nodes compared to GT nodes (GT action list is a reference, not the only ground truth)
   - A node is considered completely correct if:
     * Skills are exactly identical (after normalization) — skill match remains STRICT
     * Objects and parameters adopt FLEXIBLE equivalence in Standard Mode: judge semantic/category/part–whole/alias equivalence and functional validity by combining the first image and the task summary (or goal inferred from the DAG/GT). If the model’s node would effectively achieve the same effect in the shown scene, treat it as a match even if it differs from the GT label/argument
     * Function parameters are appropriate overall (after normalization); minor deviations are acceptable if they do not change the action’s functional effect under the current scene
   - In CSS Mode (numeric object IDs), object references must be strictly identical (no relaxation applies)
   - Analysis process must clearly indicate which specific nodes match and calculate exact proportion
   - **Scoring rule: Round DOWN the proportion to nearest integer out of 10** (use both `floor(Z*10)` and `(Y*10)//X` and ensure they are identical; if they differ, use `(Y*10)//X`)
   - Example: If 2 out of 3 nodes match (2/3 = 0.66), score = 6 points (rounded down from 6.6)
   - If no nodes match, score = 0 points
   - Object equivalence and function parameters need to be judged based on actual conditions. For example, grasp(drawer) and grasp(drawer_handle) should be considered the same function because grabbing a drawer is actually grabbing the handle; pull(drawer_handle, outward) and pull(drawer, open) should also be considered the same because pulling the drawer handle outward actually opens the drawer.

2. **Task completion degree (0-10 points)**:
   - **As a world simulator**, analyze the key object state changes required for task completion
   - Focus on **object states, not robot arm movements**
   - Based on the first image, GT action list, and DAG, identify all critical object state changes needed (GT action list serves as a reference; alternative reasonable action sequences that achieve the same critical states should be counted)

   **Critical State Identification Rules**:
   - Only count **manipulated object state changes**, not robot arm actions
   - Actions like `move_to`, `grasp`, `hold`, `approach`, `align`, `pregrasp`, `release`, `retract`, `look`, `scan`, `plan`, `think` affect robot arm or sensing, **NOT** object states
   - Only actions that **physically change object properties** count as state changes
   - Examples: `turn(tap, direction, position)` → tap angle changes; `place(apple, bowl)` → apple location/support changes; `pick_up(obj)` only counts if support changes from surface to gripper (i.e., object leaves previous support)

   **Simulation Process with Constraints**:
   - **Flexible equivalence principle (Standard Mode)**: If the model uses actions/objects/parameters that differ from the GT but are functionally reasonable given the image and task summary, and they lead to the same critical state change, count them as achieved. This relaxation does NOT apply in CSS Mode where object IDs must match exactly
   - **Visual constraint analysis**: Carefully examine the first image for spatial layouts, obstacles, object positions, and environmental constraints that affect execution feasibility
     * Example: If task is "open microwave door" but image shows a bowl blocking the door, this creates an obstacle constraint
     * Consider reachability, collision avoidance, spatial relationships, and object accessibility
   - **Physics-based realistic simulation**: Ensure all simulated actions follow real-world physics laws
     * Objects cannot pass through solid barriers, gravity affects object placement, force requirements for manipulation
     * Consider realistic manipulation constraints (size, weight, fragility, stability)
   - **Step-by-step execution simulation**: Process model's plan sequentially with visual and physical constraints
   - **Skill continuity check**: If skills are not coherent/logical, execution fails
   - **Dependency enforcement**: If a critical state is NOT achieved, all subsequent states that depend on it cannot be achieved
   - **DAG utilization**: Use DAG to identify parallel vs. sequential state dependencies
     * Parallel branches (e.g., apple→bowl, banana→bowl, pear→bowl) are independent
     * Sequential dependencies must be satisfied in order
   - **State tracking**: Only count achieved states that are actually reachable given previous failures and constraints

   **Scoring Examples**:
   - Turn on tap: 1 critical state (tap rotated to open position). If model uses wrong direction → 0/1 = 0 points
   - Put fruits in bowl: 3 parallel critical states (apple→bowl, banana→bowl, pear→bowl). If model achieves 2 → 2/3 = 0.66 → 6 points
   - If early step fails and blocks later steps, later states don't count even if planned correctly

   **Scoring rule: Round DOWN the achievement proportion to nearest integer out of 10**
   - Calculate: (Achieved critical states / Total critical states) * 10, then floor()
   - Also compute `(Achieved * 10) // Total` and ensure both give the same integer; if not, use the integer-division result
   - If no critical states are achieved, score = 0 points

### Output format
Please output your evaluation results according to the following structure, 
and you need to output the detailed analysis step by step in reason part.
```json
{{
    "node_correctness" : {{"reason": "Detailed analysis: List which specific nodes match, calculate exact proportion, explain scoring logic. Format: 'GT nodes: [list]. Model nodes: [list]. Matching nodes: [list]. Total GT nodes: X. Correct model nodes: Y. Proportion: Y/X = Z. Score: floor(Z*10) = W; (Y*10)//X = W_alt; Final score = W (must equal W_alt). Also include an Object Matching Table and a one-to-one node matching table.'", "result": x}},
    "task_completion_degree" : {{"reason": "Detailed world simulation analysis: 1) Visual constraint analysis from image. 2) Identify critical object state changes (exclude robot arm actions). 3) Physics-based realistic simulation. 4) Use DAG for dependencies. 5) Step-by-step simulation with constraints. 6) Rollout trajectory table with state deltas. Format: 'Visual constraints: [analysis]. Critical object states: [list with dependencies and thresholds]. Physics simulation + rollout: [step-by-step analysis with preconditions, effects, and state delta]. Reachable states: [list]. Total critical states: X. Achieved states: Y. Proportion: Y/X = Z. Score: floor(Z*10) = W; (Y*10)//X = W_alt; Final score = W (must equal W_alt).’", "result": y}},
    "planning_issue_analysis": {{"issue_types": ["list of issue categories"], "detailed_analysis": "Comprehensive categorization and analysis of planning problems found in the model's plan. Categories may include: wrong_object, wrong_order, missing_steps, impossible_actions, constraint_violations, physics_violations, spatial_reasoning_errors, parameter_mismatch, aliasing_errors, CSS_mode_reference_errors, etc. Provide specific examples and explanations for each identified issue type."}},
    "comprehensive_evaluation": "Give an overall evaluation as a world simulator, point out the highlights and shortcomings of the model planning performance, and give actionable suggestions for improvement."
}}
```

The data I provide is as follows:
{data}
Please output your results as required.

### **Final Guidelines**  
- **Automatically detect evaluation mode**: Analyze the action functions in the input data to determine whether to use Standard Mode or CSS Mode.
- **Object reference requirements**:
  - **Standard Mode**: Objects can be referred to by textual descriptions (e.g., pick_up(apple), move_to(table)).
  - **CSS Mode**: Objects must be referred to by their serial number as shown in the image (e.g., pick_up(3) instead of pick_up(apple)). In CSS Mode, any textual object name counts as a mismatch.

**Canonicalization & Matching (MANDATORY for Node Correctness)**  
- **Skill normalization**: lowercase; strip spaces/underscores/hyphens (turn_on, turn-on, turn on → turnon). Do not infer new skills (turn ≠ rotate unless GT says so).
- **Object semantic match**:
  - **Standard Mode (flexible)**: Build an Object Matching Table listing each GT object, its canonical label, accepted aliases, and the model object after canonicalization, then mark match / no-match using flexible equivalence judged by the image and task summary/goal. Accept category-level, alias, and part–whole substitutions if they are functionally equivalent in the scene (e.g., drawer_handle~drawer when pulling to open). Minor parameter deviations are acceptable if they do not change the intended effect.
  - **CSS Mode (strict)**: IDs must be identical (GT 3 ↔ Model 3). Non-ID mention → mismatch. No relaxation for objects.
  - **Alias rules (required, non-exhaustive)**: tap ~ faucet, fridge ~ refrigerator, sofa ~ couch, trash_can ~ bin, cup ~ mug, pan ~ frying_pan, pot ~ saucepan, phone ~ cellphone.
Affordance-critical adjectives must match: red_button ≠ green_button when color selects function; left_door ≠ right_door; upper_drawer ≠ lower_drawer.
  - Plural/singular and determiners are ignored only if quantity is not functionally critical.
- **Parameter normalization & flexibility**: Directions: counterclockwise ~ anticlockwise ~ ccw; clockwise ~ cw. Binary states: (open, closed), (on, off), (locked, unlocked). Spatial relations: in/inside/into, on/onto, under/below, front_of/in_front_of, behind/back_of. Angles/quantities: exact if specified; if GT range → model value must lie within range; if GT is nominal (“fully_open”), model must specify equivalent nominal. In Standard Mode, minor parameter divergences (granularity/angle/target region) are acceptable if they would not change the functional outcome under the scene constraints. In CSS Mode, maintain strict parameter consistency when it affects object identity.
 - One-to-one best matching between model nodes and GT nodes; no partial credit per node.
 - Show arithmetic with both floor(Z*10) and (Y*10)//X in the “reason”.

**State Rollout Engine (MANDATORY) — Strict sequential execution & scoring**  
You must evaluate via “plan steps → ordered simulation → state delta ledger”. All task completion scores must be derived from states actually achieved during rollout; do not skip steps or guess.
- **A. Object State Machine (per manipulated object)**: Maintain: pose/support/contact, orientation/angle, openness ∈ (closed, partially_open(f∈[0,1]), fully_open), containment/attachment, activation, integrity/assembly.
Robot-only actions never directly modify object state. pick_up(obj) counts only if support changes off the previous surface (lifted).
- **B. Skill→Effects Matrix (use these rules)**:
| Skill (normalized)                     | Preconditions (must pass)                                                   | Physical/contact rules                                                     | Effects on state (if success)                                                                         |
| -------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `push(drawer, dir, dist/force?)`       | Reachable; `dir` aligns with slide direction (outward/forward); no blocking | Force > static friction; travel ≤ remaining stroke; truncated by obstacles | `openness` increases; if open\_fraction ≥ threshold → `fully_open` or `partially_open(f)`             |
| `pull(drawer, dir, dist)`              | As above (pull outward)                                                     | As above                                                                   | Same as above (increase open\_fraction)                                                               |
| `push(door, dir)`                      | Act on **non-hinge** side; openable direction; not latched/blocked          | Torque > hinge resistance; stop at obstacle                                | Increase door angle; if ≥ threshold → open                                                            |
| `turn(knob/tap, cw/ccw, angle/target)` | Reachable; free rotation space                                              | Angle limited by endstop                                                   | Update angle; if passes “on/open” target → activation/open changes                                    |
| `place(obj, receptacle/area)`          | Object under control; valid support                                         | Stable placement; size fits                                                | Update `pose/support`; if into container → `containment: in`                                          |
| `pick_up(obj)`                         | Grasp feasible                                                              | Must **lift** off prior support                                            | If lifted → support becomes `gripper` (state change). “Clamp only” without lift → **no state change** |

Default thresholds (use unless GT/DAG specifies otherwise):
- Drawer open if open_fraction ≥ 0.3 of max travel.
- Door open if angle ≥ 20°.

- **C. Visual & Physical Constraints (fine rules)**: Directions: counterclockwise ~ anticlockwise ~ ccw; clockwise ~ cw.
- Reachability/occlusion gates precondition; “move_to/approach” can enable reachability but does not change states.
- Direction correctness: drawer “forward/outward” to open; “inward” closes.
- Obstacle/collision truncates effective travel; if below threshold, not “opened”.
- Force: if text says “light touch/insufficient force”, treat as failure; otherwise assume reasonable force.

- **D. Rollout Ledger (record every step)**
For each plan step:
- Normalize skill; check preconditions (reachability/obstacle/direction/force).
- If pass, apply Effects Matrix; compute state delta (before→after).
- Log to Rollout Trajectory Table: step id, original & normalized action, precondition pass/fail + reason, state delta fields, numeric open_fraction/angle if relevant, and which critical states were newly satisfied.
- Enforce dependencies from DAG: dependent states only unlock after prerequisites achieved.
- If later steps reverse earlier achieved states (e.g., closing a door), final set reflects end state; each critical state counts at most once.
Rollout-first rule (hard): The achieved count Y for scoring must be computed only from states marked as achieved in the Rollout Trajectory Table.
If a step like push(drawer, forward, 0.4*d_max) passes and reaches threshold, the drawer opened state must be counted (not 0).

- **E. Drawer/Door special rules (prevent false zero)**
- Compute effective travel d_eff = min(requested, remaining, clearance_limit); update open_fraction.
- If direction wrong (inward when opening), no open progress.
- If blocked, d_eff ≈ 0 → no open.
- If opened then closed later, final “opened” is not counted.

- **F. Scoring (redundant integer check)**
Let X = total critical states, Y = number achieved after rollout.
Compute Z = Y/X; Score = floor(Z*10) and Score_alt = (Y*10)//X. They must be equal; otherwise use Score_alt.

- **G. Required artifacts in “reason”**
- Node correctness: Object Matching Table; node matching table; both scoring formulas shown.
- Task completion: Visual constraints; full list of critical states with thresholds & dependencies; Rollout Trajectory Table (summary) showing preconditions/effects/state deltas and which critical states were hit; final reachable states; both scoring formulas shown.

**Few-shot anchors (follow exactly)**:
- A — Tap rotation (Standard Mode)
GT: turn(tap, ccw, open_angle)
Model: turn(faucet, anticlockwise, open_angle) → skill match ✅; object alias ✅; params normalized ✅ → Y=1, X=1 → score 10 (floor & // both 10).
- B — Fruits to bowl (3 parallel)
Critical states: apple→in(bowl), banana→in(bowl), pear→in(bowl); model achieves 2 → Y=2, X=3 → score 6.
- C — grasp is NOT a state change
Only move_to, grasp, hold → Y=0 → score 0.
- D — CSS Mode strictness
GT uses IDs: pick_up(3), place(3,5); model uses pick_up(cup) → object reference mismatch → node incorrect.
- E — Drawer push opens (must count)
GT critical: drawer→opened (≥0.3); plan: push(drawer, forward, 0.4*d_max); preconditions pass → open_fraction 0→0.4 ≥ 0.3 → count as achieved → Y=1, X=1 → score 10.

 - F — Flexible equivalence (Standard Mode, image-supported; counts for both node correctness and task completion)
   Use the first image and task summary to judge functional equivalence. If actions/objects/parameters differ but lead to the same effect in the scene, accept as match (does NOT apply in CSS Mode):
   1) Task summary: "<closing a drawer>"; Model: push(drawer, closed_position); GT: push(drawer, to_the_cabinet) → functional effect: drawer closed → Accept ✅
   2) Task summary: "<open a microwave door>"; Model: pull(microwave_handle, right_side_of_microwave); GT: pull(microwave_handle, open) → functional effect: door opened → Accept ✅
   3) Task summary: "<closing a drawer>"; Model: push(drawer, closed_position); GT: push(drawer, close) → functional effect: drawer closed → Accept ✅
   4) Task summary: "<opening a drawer>"; Model: pull(drawer_handle, outward); GT: pull(drawer, open) → part–whole equivalence (handle~drawer) with same effect (opened) → Accept ✅
   5) Task summary: "<turn off a desk lamp>"; Model: press(none, power_button); GT: press(lamp_switch) → alias/equivalent control leads to lamp off → Accept ✅
   6) Task summary: "<opening a drawer>"; Model: pull(drawer_handle, front_of_cabinet); GT: pull(drawer_handle, open) → direction/target phrasing differs, effect same (opened) → Accept ✅

**Critical World Simulation Reminders**:
- **ESSENTIAL: Visual constraint analysis**: The first image provides critical initial conditions and constraints (obstacles, spatial layouts, object positions) that MUST be considered in simulation
- **ESSENTIAL: Physics-based realistic simulation**: All actions must follow real-world physics laws (gravity, collision, force, stability)
- **Distinguish robot actions vs. object state changes**: `move_to`, `grasp`, `hold` do NOT change object states
- **Only count manipulated object property changes**: position, orientation, state (open/closed), etc.
- **Enforce strict dependency chains**: Failed early states block dependent later states
- **Use DAG for dependency understanding**: Parallel branches are independent, sequential branches have dependencies
- **Simulate execution realistically**: Consider skill continuity and logical action sequences with visual/physical constraints
- **No partial credit for unreachable states**: If prerequisite fails, dependent states score 0
- **MANDATORY: Issue categorization**: After simulation, comprehensively analyze and categorize all planning problems found

- **Please strictly follow the output format. The output must be in json format.**
- **When evaluating, pay special attention to the correspondence between the model output steps and the DAG.**
- **Please provide detailed analysis with specific numbers and proportions for each evaluation criterion.**
- **Please conduct the above evaluation objectively and in detail.**
- **A grading standard for task_completion_degree**: In the list of actions that physically change object properties, if a preceding action is not completed, subsequent actions cannot be considered achieved.
- **A grading standard for task_completion_degree**: Object equivalence and function parameters need to be judged based on actual conditions. For example, grasp(drawer) and grasp(drawer_handle) should be considered the same function because grabbing a drawer is actually grabbing the handle; pull(drawer_handle, outward) and pull(drawer, open) should also be considered the same because pulling the drawer handle outward actually opens the drawer.

Please output your evaluation results strictly according to the following structure:
```json
{{
    "node_correctness" : {{"reason": "Detailed analysis: List which specific nodes match, calculate exact proportion, explain scoring logic. Format: 'GT nodes: [list]. Model nodes: [list]. Matching nodes: [list]. Total GT nodes: X. Correct model nodes: Y. Proportion: Y/X = Z. Score: floor(Z*10) = W; (Y*10)//X = W_alt; Final score = W (must equal W_alt). Also include an Object Matching Table and a one-to-one node matching table.'", "result": x}},
    "task_completion_degree" : {{"reason": "Detailed world simulation analysis: 1) Visual constraint analysis from image. 2) Identify critical object state changes (exclude robot arm actions). 3) Physics-based realistic simulation. 4) Use DAG for dependencies. 5) Step-by-step simulation with constraints. 6) Rollout trajectory table with state deltas. Format: 'Visual constraints: [analysis]. Critical object states: [list with dependencies and thresholds]. Physics simulation + rollout: [step-by-step analysis with preconditions, effects, and state delta]. Reachable states: [list]. Total critical states: X. Achieved states: Y. Proportion: Y/X = Z. Score: floor(Z*10) = W; (Y*10)//X = W_alt; Final score = W (must equal W_alt).’", "result": y}},
    "planning_issue_analysis": {{"issue_types": ["list of issue categories"], "detailed_analysis": "Comprehensive categorization and analysis of planning problems found in the model's plan. Categories may include: wrong_object, wrong_order, missing_steps, impossible_actions, constraint_violations, physics_violations, spatial_reasoning_errors, parameter_mismatch, aliasing_errors, CSS_mode_reference_errors, etc. Provide specific examples and explanations for each identified issue type."}},
    "comprehensive_evaluation": "Give an overall evaluation as a world simulator, point out the highlights and shortcomings of the model planning performance, and give actionable suggestions for improvement."
}}
```
"""

########################
# Extract the next step from this response.
########################
PROMPT_TEMPLATE_EXTRACT_STEP = """
Extract the next step from this response. The step should be in the format: skill(element1, element2, ...)

For example:
- "grasp(microwave_handle)"
- "push(microwave_handle, close)"
- "move_to(none, drawer)"

Response: {response}

Return ONLY the step in the correct format, nothing else.
"""

########################
# Compare the extracted step with the ground truth step.
########################
PROMPT_TEMPLATE_COMPARE_STEPS = """
You are evaluating the similarity between two robot action steps. Your task is to evaluate the extracted step against the ground truth step using specific criteria.

### Input data format
{{
    "extracted_step": "The step extracted from model response",
    "gt_step": "The ground truth step"
}}

### Evaluation criteria

1. **Skill usage accuracy (0, 1 point)**:
   - Consider only the skill/action part of both steps (ignore the objects/parameters)
   - Score based on whether the skills are completely identical.
   - 1: Perfect skill match (exactly the same action word)
   - 0: Different or not completely identical skills

2. **Operation object reasonableness (0, 0.5, or 1 point)**:
   - Consider only the operation object(s) part of both steps
   - Check if objects refer to similar or related things
   - 1: Objects are identical or clearly refer to the same thing (e.g., "door_handle" vs "door handle", "microwave_door" vs "microwave_handle")
   - 0.5: Objects are similar or related (e.g., "table" vs "table_leg", "cup" vs "mug", objects referring to the same category)
   - 0: Objects are completely different or unrelated

3. **Parameter accuracy (0, 0.5, or 1 point)**:
   - Consider the function parameters and their quality
   - If skill is 0 or object is 0, this parameter score is automatically 0
   - If skill and object both have scores, evaluate the parameter quality:
   - 1: Parameters are completely correct and high quality
   - 0.5: Parameters are partially correct or medium quality
   - 0: Parameters are incorrect or low quality

### Evaluation Guidelines
- Skill evaluation must be strict - only award 1 point when skills are exactly identical
- Object evaluation can be more flexible - consider semantic similarity
- Parameter evaluation depends on skill and object scores being non-zero
- Give detailed reasons explaining your scoring decision
- Consider the context and precision required for robot task execution

### Output format
Please output your evaluation results strictly according to the following JSON structure:
{{
    "skill_usage_accuracy": {{"result": x, "reason": "brief explanation of skill evaluation"}},
    "operation_object_reasonableness": {{"result": y, "reason": "brief explanation of object evaluation"}},
    "parameter_accuracy": {{"result": z, "reason": "brief explanation of parameter evaluation"}}
}}

The data I provide is as follows:

Extracted step: {extracted_step}
Ground truth step: {gt_step}

Please output your results as required.
"""

########################
# Extract ONLY the yes/no answer from this response.
########################
PROMPT_TEMPLATE_EXTRACT_YES_NO = """
Extract ONLY the yes/no answer from this response.
Return ONLY "yes" or "no" with no other text or explanation.

Response: {response}
"""

########################
# Other templates
########################
PROMPT_TEMPLATE_NORMALIZE_CHOICE = """
Extract ONLY the final multiple choice answer(s) from this response.
Return ONLY the letter(s) A,B,C,D with no other text or explanation.
If multiple answers, separate with commas.
Response: {answer}
"""

PROMPT_TEMPLATE_POINT = """
You are given a coordinate string that represents a point (x, y) in various possible formats.
Your task is to extract and normalize it to the standard format (x, y) where x and y are numbers.

Examples:
- Input: "(0.5, 0.7)" -> Output: "(0.5, 0.7)"
- Input: "[0.3, 0.8]" -> Output: "(0.3, 0.8)"
- Input: "The point is at 0.2, 0.9" -> Output: "(0.2, 0.9)"
- Input: "x=0.1, y=0.6" -> Output: "(0.1, 0.6)"
- Input: "coordinates: (0.4, 0.2)" -> Output: "(0.4, 0.2)"

Return ONLY the normalized coordinate in format (x, y) with no additional text.
If you cannot find valid coordinates, return "(0.0, 0.0)".

Input coordinate string: {coord_string}
"""

PROMPT_TEMPLATE_TRAJECTORY = """
You are given a trajectory string that represents a sequence of points (x, y) in various possible formats.
Your task is to extract and normalize it to the standard format [[x1,y1], [x2,y2], ...] where x and y are numbers.

Examples:
- Input: "[[0.5, 0.7], [0.6, 0.8]]" -> Output: [[0.5, 0.7], [0.6, 0.8]]
- Input: "[(0.3, 0.8), (0.4, 0.9)]" -> Output: [[0.3, 0.8], [0.4, 0.9]]
- Input: "The trajectory is: 0.2,0.9 ; 0.3,1.0 ; 0.4,1.1" -> Output: [[0.2, 0.9], [0.3, 1.0], [0.4, 1.1]]
- Input: "points: x=0.1,y=0.6 ; x=0.2,y=0.7" -> Output: [[0.1, 0.6], [0.2, 0.7]]
- Input: "Trajectory: [(0.4, 0.2), (0.5, 0.3), (0.6, 0.4)]" -> Output: [[0.4, 0.2], [0.5, 0.3], [0.6, 0.4]]

Return ONLY the normalized trajectory in format [[x1,y1], [x2,y2], ...] with no additional text.
If you cannot find valid trajectory, return [[0.0, 0.0]].

Input trajectory string: {trajectory_string}
"""

PROMPT_TEMPLATE_EVAL_OPEN = """
You are evaluating the accuracy of a response to an open-ended question.

Ground truth answer: {gt_answer}

Response to evaluate: {response}

On a scale from 0 to 1, how accurate is the response compared to the ground truth?
- Score 1.0: The response is completely correct and covers all key points in the ground truth.
- Score 0.5: The response is partially correct but misses important details or has significant inaccuracies.
- Score 0.0: The response is completely incorrect or unrelated to the ground truth.

Provide your evaluation in the following JSON format:
{{
    "score": <score as a float between 0 and 1>,
    "explanation": "<brief explanation of your scoring>"
}}
"""
