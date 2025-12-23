# VLM-Guided RL Training: Beyond Heuristics

## Problem: Your RL Agent Relies Too Much on Heuristics

After 6 hours of training, your RL agent has learned to skip the title screen and name input, but it's **spamming A in dialogue** instead of learning when A-presses are actually needed. This is because:

1. **Hard-coded dialogue override** in `run_rl.py` forces A after 3 steps of dialogue
2. **No reasoning about WHY agent is stuck** - just simple coordinate/dialogue counting
3. **RL never learns to handle dialogue naturally** - the heuristic does it for them
4. **Fixed reward weights** don't adapt to game context

## Solution: VLM-Guided RL with Minimal Heuristics

Inspired by **PokemonRedExperiments** but enhanced with VLM reasoning, this approach uses:

### 1. VLM Unstuck Detection (`agent/vlm_unstuck.py`)

Instead of hard-coded rules like "if coordinates repeat 5 times, stuck", the VLM:
- **Analyzes game state with context**: "Agent is in a room with NPCs but not talking to them"
- **Identifies stuck patterns**: "coordinate_loop", "dialogue_spam", "menu_trap"
- **Provides reasoning**: "Agent is walking in circles near the exit but not taking it"
- **Suggests exploration**: "Try pressing UP to reach the door"

```python
from agent.vlm_unstuck import VLMUnstuckDetector

# Initialize VLM unstuck detector
vlm_unstuck = VLMUnstuckDetector(
    vlm=vlm,
    check_frequency=30,  # Check every 30 steps
    enable_suggestions=True
)

# During training
vlm_stuck_pattern = vlm_unstuck.check_stuck(frame, game_state)
if vlm_stuck_pattern:
    # VLM detected stuck with confidence and reasoning
    # Apply penalty based on VLM's analysis
    reward += vlm_stuck_pattern.reward_modifier
```

### 2. VLM-Guided Reward Shaping

Instead of fixed weights (exploration=1.0, dialogue=5.0), VLM dynamically adjusts based on **what's important NOW**:

```python
from agent.vlm_unstuck import VLMGuidedRewardShaper

reward_shaper = VLMGuidedRewardShaper(vlm, check_frequency=100)

# VLM analyzes: "Agent is in starting town, should explore more"
# Updates priorities: exploration=2.0, dialogue=0.5

# Reward gets scaled by priority
exploration_reward = base_reward * reward_shaper.current_priorities['exploration']
```

### 3. Reduced Dialogue Override (Let RL Learn!)

**Before** (heavy heuristic):
```python
# Force A after just 3 steps of dialogue
if dialogue_active and steps >= 3:
    action = 'A'  # Override RL decision!
```

**After** (minimal heuristic):
```python
# Only force A after 10+ steps (emergency fallback)
# Give RL 3x more chances to learn naturally
if dialogue_active and steps >= 10:
    action = 'A'  # Emergency override only
    logger.info("RL should have learned this!")
```

### 4. Bonus Rewards for Following VLM Advice

When VLM suggests an action and it works:
```python
if agent_followed_suggestion and agent_got_unstuck:
    reward += 2.0  # Big bonus!
    # RL learns: "VLM suggestions are valuable"
```

## Architecture Comparison

### Old Heuristic Approach
```
Game State → Hard Rules → Force Action
  ↓
"Dialogue detected" → Press A (no learning)
"Coordinates repeat" → Add penalty (no context)
```

### New VLM-Guided RL Approach
```
Game State → VLM Reasoning → Context + Suggestion
  ↓                              ↓
RL Policy → Consider Context → Choose Action
  ↓
Reward shaped by VLM priorities + Bonus for good suggestions
  ↓
RL learns to USE VLM reasoning (true intelligence!)
```

## Key Differences from PokemonRedExperiments

PokemonRedExperiments uses:
- Coordinate novelty (knn index on frames)
- Event flags for progress
- Simple loop detection

We enhance this with:
- ✅ **VLM analyzes WHY stuck** (not just that it's stuck)
- ✅ **Dynamic reward priorities** (VLM adjusts what matters)
- ✅ **Contextual suggestions** (VLM guides exploration)
- ✅ **Learning from VLM advice** (bonus rewards)
- ✅ **Minimal hard overrides** (RL learns true behavior)

## Usage

### Train with VLM Guidance (Recommended)

```bash
# Set API key for your VLM backend
export OPENAI_API_KEY="your-key"
# or ANTHROPIC_API_KEY for Claude
# or GOOGLE_API_KEY for Gemini

# Train with VLM unstuck guidance
python train_rl_vlm_guided.py \
    --timesteps 10000000 \
    --vlm_backend qwen \
    --vlm_unstuck_freq 30 \
    --vlm_guidance_freq 100
```

### Resume Your Existing Training (Add VLM)

```bash
# Resume your 6-hour checkpoint with VLM guidance added
python train_rl_vlm_guided.py \
    --resume .rl_checkpoints/pokemon_emerald_rl_XXXXX_steps.zip \
    --timesteps 5000000 \
    --vlm_backend qwen
```

### Run Agent with Reduced Heuristics

```bash
# The modified run_rl.py now only overrides after 10 steps (was 3)
python run_rl.py \
    --checkpoint .rl_checkpoints/your_checkpoint.zip \
    --vlm_backend qwen \
    --render
```

## Expected Improvements

After training with VLM guidance:

1. **Dialogue Handling**: Agent learns WHEN to press A, not just spam
   - VLM detects: "Dialogue present but not advancing"
   - Suggests: Press A
   - RL learns: A in dialogue is good
   - Result: Natural dialogue progression

2. **Smart Exploration**: Agent explores purposefully
   - VLM: "New area to the north, explore it"
   - RL adjusts policy to favor UP
   - Result: Efficient exploration

3. **Unstuck Without Heuristics**: Agent escapes loops naturally
   - VLM detects: "Walking in circles near door"
   - Suggests: "Try moving toward door"
   - RL learns pattern
   - Result: Self-unsticking behavior

4. **Context-Aware Rewards**: Priorities adapt to game progress
   - Early game: High exploration reward
   - Story moment: High dialogue reward
   - Result: Always doing what matters most

## Monitoring VLM Impact

During training, watch for:

```
🚨 VLM detected stuck: coordinate_loop
   Confidence: 0.85
   Reasoning: Agent walking in circles in starting room
   Suggestion: Try pressing UP toward door

✅ VLM suggestion worked! Bonus: +2.0
   Agent successfully escaped stuck pattern

🎯 Updated reward priorities: {'exploration': 2.0, 'dialogue': 0.5, 'events': 1.0}
```

## Files Modified/Created

1. **`agent/vlm_unstuck.py`** (NEW): VLM unstuck detection and reward shaping
2. **`agent/rl_trainer.py`** (MODIFIED): Integrated VLM unstuck and reward shaping
3. **`run_rl.py`** (MODIFIED): Reduced dialogue override threshold (3→10 steps)
4. **`train_rl_vlm_guided.py`** (NEW): Training script with VLM features enabled

## Philosophy

**The Goal**: Train an RL agent that UNDERSTANDS the game, not just memorizes heuristics.

- ❌ **Bad**: Hard-code "if dialogue, press A"
- ✅ **Good**: Reward agent for advancing dialogue, let it learn A is the tool

- ❌ **Bad**: "If stuck 5 times, random action"
- ✅ **Good**: VLM analyzes why stuck, suggests strategy, RL learns pattern

- ❌ **Bad**: Fixed reward weights forever
- ✅ **Good**: VLM updates priorities as game progresses

This creates an agent that can handle **novel situations** because it learned the underlying logic, not just shortcuts.

## Next Steps

1. **Resume your 6-hour training with VLM guidance**:
   ```bash
   python train_rl_vlm_guided.py --resume <your_checkpoint> --vlm_backend qwen
   ```

2. **Monitor VLM detections** to see where agent gets stuck

3. **Watch RL learn to use VLM context** (bonus rewards will increase)

4. **Compare behavior**: Old heuristic spam vs. new learned behavior

5. **Iterate**: VLM will identify new stuck patterns, helping RL improve

## Troubleshooting

**"VLM is slow!"**
- Adjust `--vlm_unstuck_freq` to check less often (e.g., 50 or 100)
- Use faster VLM backend (qwen is fast, claude is slow but smart)

**"Agent still spamming A!"**
- Check that VLM unstuck is enabled: `use_vlm_unstuck=True`
- Verify dialogue reward is high enough to encourage learning
- May need more training steps for behavior to change

**"VLM not detecting stuck patterns"**
- Lower confidence threshold in `VLMUnstuckDetector`
- Check VLM responses in logs
- Try different VLM backend (claude is better at reasoning)

**"Training is slower with VLM"**
- Expected! VLM adds compute overhead
- But results in SMARTER agent
- Can disable VLM after learning stabilizes

## References

- **PokemonRedExperiments**: Coordinate novelty, frame knn, event rewards
- **Your Code**: Dialogue detection, memory reading, SNN features
- **This Enhancement**: VLM reasoning, dynamic rewards, minimal heuristics
