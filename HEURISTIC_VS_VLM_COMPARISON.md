# Comparison: Heuristic vs. VLM-Guided RL

## Current Problem (After 6 Hours Training)

Your RL agent has trained for 6 hours and learned to:
- ✅ Skip title screen
- ✅ Handle name input
- ❌ **Spams A button in dialogue** (heuristic does this, not RL learning)
- ❌ **Relies on hard-coded unstuck rules**

## Root Cause Analysis

### Where Heuristics Override RL

**File: `run_rl.py`, Line ~125**
```python
# DIALOGUE OVERRIDE: If dialogue present for 3+ steps, force A
if dialogue_active and self.dialogue_steps_without_a >= 3:
    rl_action = 'A'  # RL decision overridden!
```

**Problem**: RL never learns to press A in dialogue because heuristic does it first!

### Simple Loop Detection (No Context)

**File: `agent/rl_trainer.py`, Line ~534**
```python
# If unique coordinates < 5 in last 20 steps, apply penalty
if unique_coords < 5:
    reward += loop_penalty  # But WHY is it stuck?
```

**Problem**: No reasoning about WHAT is happening or HOW to fix it.

## Comparison Table

| Aspect | Current Heuristic Approach | VLM-Guided RL Approach |
|--------|---------------------------|------------------------|
| **Dialogue Handling** | Force A after 3 steps | Reward A-press in dialogue, let RL learn (override only after 10 steps) |
| **Stuck Detection** | Count coordinates < 5 unique | VLM analyzes: "Agent walking in circles near door but not entering" |
| **Unstuck Strategy** | Random penalty | VLM suggests: "Try pressing UP toward door" + reward if it works |
| **Reward Weights** | Fixed forever | VLM adjusts: "Early game = high exploration, story moment = high dialogue" |
| **Learning** | RL learns workarounds | RL learns TRUE game behavior |
| **Generalization** | Fails on novel situations | Handles new scenarios (learned reasoning) |
| **Interpretability** | Black box + hard rules | VLM provides reasoning logs |

## Detailed Example: Dialogue Scenario

### Scenario: NPC blocks path, says "Welcome to Pokemon!"

#### Current Heuristic Approach
```
Step 1: RL policy predicts 'UP' (blocked by NPC)
Step 2: RL policy predicts 'UP' (still blocked)
Step 3: RL policy predicts 'RIGHT' (exploring)
Step 4: HEURISTIC OVERRIDE → Force 'A' (dialogue detected 3 steps ago)
Step 5: Dialogue advances, NPC moves

Reward breakdown:
- No reward for A-press (heuristic did it, not RL)
- Small dialogue progress reward (but disconnected from action)
- RL learns: "Being near NPCs eventually makes progress"
- RL NEVER learns: "I should press A when NPCs talk to me"
```

**Result**: Agent spams A everywhere hoping it triggers progress (like your current behavior!)

#### VLM-Guided RL Approach
```
Step 1: RL policy predicts 'UP' (blocked by NPC)
Step 2: RL policy predicts 'UP' (still blocked)
Step 3: RL policy predicts 'A' (learning from dialogue reward)
Step 4: Dialogue advances!

VLM Analysis (every 30 steps):
"Agent encountered NPC blocking path. Dialogue initiated. Agent pressed A to advance dialogue. This is correct behavior."

Reward breakdown:
- RL pressed A → gets dialogue progress reward (+5.0)
- VLM detected good pattern → suggestion bonus (+2.0)
- Clear cause-effect: "A in dialogue = reward"
- RL learns: "Press A when dialogue appears"
```

**Result**: Agent learns WHEN and WHY to press A (not just spam it everywhere).

## Detailed Example: Stuck in Room

### Scenario: Agent stuck in starting room, walking in circles

#### Current Heuristic Approach
```
Steps 1-20: Walk UP, DOWN, LEFT, RIGHT randomly
- Coordinate history: [(5,5), (5,6), (5,5), (5,6), (5,5)...]
- unique_coords = 2 (less than 5)
- Apply loop_penalty = -1.0

RL receives:
- "Some mysterious penalty appeared"
- No context on WHY
- No suggestion on HOW to escape
- Hopes random exploration eventually works
```

**Result**: Agent eventually escapes by luck, but didn't learn the pattern.

#### VLM-Guided RL Approach
```
Steps 1-20: Walk UP, DOWN, LEFT, RIGHT randomly

Step 30: VLM checks for stuck
VLM Analysis:
"Agent is in starting room with door at coordinates (5, 3). 
Agent walked between (5,5) and (5,6) repeatedly but never 
approached door. This is coordinate_loop stuck pattern.
Confidence: 0.85
Suggestion: Try moving UP toward door at (5,3)"

Reward:
- Apply stuck penalty with context: -0.5 (VLM determined severity)
- RL continues exploring

Step 35: RL tries 'UP', 'UP', 'UP' (learning from exploration reward)
Step 38: Reaches door, transitions to new map!

VLM detects unstuck:
- Agent followed directional guidance
- Successfully reached new area
- Reward bonus: +2.0

RL learns:
- "When stuck in loop, trying different directions = good"
- "Moving toward unexplored areas = reward"
- "VLM suggestions are valuable"
```

**Result**: Agent learned a STRATEGY (move toward unexplored), not just memorized one room.

## Code Changes Summary

### 1. New VLM Unstuck Module (`agent/vlm_unstuck.py`)
- 350 lines of VLM reasoning logic
- `VLMUnstuckDetector`: Analyzes stuck patterns
- `VLMGuidedRewardShaper`: Dynamic reward priorities

### 2. Modified RL Environment (`agent/rl_trainer.py`)
- Integrated VLM unstuck detector
- VLM-shaped reward calculation
- Bonus rewards for following VLM suggestions
- Fallback to simple heuristics only if VLM unavailable

### 3. Reduced Heuristic Override (`run_rl.py`)
- Dialogue override: 3 steps → 10 steps
- Gives RL more chances to learn
- Only emergency fallback, not primary mechanism

### 4. New Training Script (`train_rl_vlm_guided.py`)
- Enables VLM features by default
- Configurable VLM check frequencies
- Monitoring for VLM impact

### 5. Quick Resume Script (`resume_training_with_vlm.sh`)
- Finds your latest checkpoint
- Adds VLM guidance to existing training
- Easy to run: `./resume_training_with_vlm.sh`

## Expected Training Timeline

### Continuing from Your 6-Hour Checkpoint

**First 1-2 hours with VLM:**
- VLM starts detecting stuck patterns
- Agent still relies on old behaviors
- Gradual improvement in dialogue handling

**After 3-4 hours with VLM:**
- Agent learns to press A in dialogue naturally
- Exploration becomes more purposeful
- Fewer stuck situations

**After 6-8 hours with VLM:**
- Agent handles dialogue like a human
- Self-unsticks without heuristics
- Strong strategic exploration
- Adapts to new situations

## Performance Considerations

### VLM Overhead
- VLM check every 30 steps ≈ 1-2 API calls per minute
- Each call: 100-500ms depending on backend
- Total slowdown: ~5-10% training speed

**Trade-off**: Slightly slower training, but MUCH smarter agent

### Cost Estimate (for VLM API)
- Using Qwen/Gemini: ~$0.10-0.50 per hour
- Using Claude: ~$1-3 per hour
- Using GPT-4V: ~$3-10 per hour

**Recommendation**: Start with Qwen (fast and cheap) for initial training.

## How to Use

### Quick Start (Recommended)
```bash
# Set API key
export OPENAI_API_KEY="your-key"  # or appropriate key for your VLM

# Resume your 6-hour training with VLM guidance
./resume_training_with_vlm.sh
```

### Manual Control
```bash
# Resume with specific parameters
python train_rl_vlm_guided.py \
    --resume .rl_checkpoints/pokemon_emerald_rl_<timestamp>_steps.zip \
    --timesteps 5000000 \
    --vlm_backend qwen \
    --vlm_unstuck_freq 30 \
    --vlm_guidance_freq 100
```

### Testing Behavior Changes
```bash
# Run old checkpoint (with heavy heuristics)
python run_rl.py --checkpoint <old_checkpoint> --render

# Run new checkpoint (with VLM learning)
python run_rl.py --checkpoint <new_checkpoint> --render

# Compare dialogue handling:
# Old: Spams A constantly
# New: Presses A only when needed
```

## Monitoring VLM Impact

Watch training logs for:

```
🚨 VLM detected stuck: coordinate_loop
   Confidence: 0.85
   Reasoning: Agent walking in circles in starting room
   Suggestion: Try pressing UP toward door
```

And success cases:
```
✅ VLM suggestion worked! Bonus: +2.0
   Agent successfully escaped stuck pattern
```

And reward priority updates:
```
🎯 Updated reward priorities: {'exploration': 2.0, 'dialogue': 0.5, 'events': 1.0}
   Reasoning: Agent is in starting town, should explore more
```

## Success Metrics

### Before VLM Guidance (Your Current Agent)
- Dialogue: Spams A constantly
- Stuck Rate: Moderate (heuristics help)
- Exploration: Random walk
- Novel Situations: Fails (no learned behavior)

### After VLM Guidance (Expected)
- Dialogue: Presses A when appropriate
- Stuck Rate: Low (self-unsticks)
- Exploration: Purposeful (follows VLM strategies)
- Novel Situations: Handles well (learned reasoning)

## Troubleshooting

**"Agent still spamming A"**
- Need more training time (behavior takes time to change)
- Check VLM is actually running (see logs for VLM detections)
- Increase dialogue reward weight

**"VLM checks are slow"**
- Use faster backend (qwen > gemini > claude > gpt4)
- Increase check frequency (30 → 50 or 100)
- Disable during inference (only needed for training)

**"No VLM detections in logs"**
- Check API key is set correctly
- Verify VLM backend is accessible
- Try test VLM call manually

**"Training crashes with VLM"**
- Fallback: VLM errors won't crash training
- Check error logs in llm_logs/
- Verify network connectivity

## Philosophy: True Learning vs. Shortcuts

The goal is an RL agent that **understands** Pokemon, not just memorizes button sequences.

**Bad RL Agent**: "When these pixels appear, press A 10 times"
**Good RL Agent**: "Dialogue requires A to advance, press A when dialogue is present"

**Bad RL Agent**: "When stuck 5 times, try random action"
**Good RL Agent**: "I'm near a door but not entering it, let me move toward the door"

VLM guidance teaches the agent to REASON, not just react to patterns. This creates robust behavior that generalizes to new situations.

## Next Steps

1. ✅ Read this comparison document
2. ✅ Read `VLM_GUIDED_RL_README.md` for details
3. ▶️  Run `./resume_training_with_vlm.sh`
4. 👀 Monitor VLM detections in logs
5. 🎮 Test improved agent with `run_rl.py`
6. 🔄 Iterate: VLM will identify new patterns to improve
