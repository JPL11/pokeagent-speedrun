# Hybrid Architecture: k-NN Episodic Memory + SNN Temporal Reasoning

## Overview

This document describes the integration of k-NN episodic memory with the existing SNN temporal reasoning architecture. The key insight: **k-NN and SNN complement each other rather than compete**.

## Architectural Philosophy

```
VLM (Qwen2B) → High-level semantic reasoning ("Go to Route 101")
CNN → Spatial features (local vision)
SNN → Temporal dependencies (learned patterns over time)
SLAM → Geometric mapping (walkability, paths)
k-NN → Episodic recall (similarity-based memory)
```

**k-NN is NOT a replacement for SNN** - it's an augmentation:
- **SNN** = "How does this step fit into the long temporal sequence?"
- **k-NN** = "What happened last time we saw something like this?"

## k-NN Use Cases

### 1. Navigation (Path-following)

When CNN/SLAM provides ambiguous waypoints (complex town layouts, forests, repeated structures):

```python
k-NN recalls: "When we saw trees arranged like this, 
              the door was 4 tiles down and 2 left"
```

**Stored Experience:**
```python
nav_key = concat(
    cnn_features[32-dim],    # Local visual features
    story_vec[16-dim],       # Current quest context
    waypoint_hint[8-dim]     # SLAM direction hint
)
→ (action, reward, milestone_reached)
```

**Retrieved Information:**
- `suggested_actions`: k most common actions in similar states
- `action_weights`: Probability distribution over actions
- `avg_reward`: Expected reward from similar past states
- `milestone_likelihood`: Probability a milestone was reached

### 2. Battle Strategy

When the same matchup appears again:

```python
k-NN recalls: "Last time Mudkip (L8) fought Poochyena (L6),
              Water Gun was optimal"
```

**Stored Experience:**
```python
battle_key = concat(
    battle_vec[16-dim],         # Strategy features
    hp_state[4-dim],            # [my_hp_ratio, opp_hp_ratio, counts]
    type_advantages[8-dim]      # Type matchup scores
)
→ (action, outcome, gym_battle_flag)
```

**Retrieved Information:**
- `suggested_move`: Most common successful action
- `win_rate`: Success rate in similar situations
- `risk_score`: Probability of loss/faint

### 3. Quest Memory

When similar dialogue patterns occur:

```python
k-NN recalls: "This NPC dialogue pattern previously led to
              milestone X being completed"
```

### 4. Stuck Detection

When SLAM suggests a path but agent is stuck:

```python
k-NN retrieves similar "stuck situations"
→ suggests "try moving left around wall"
```

## Integration with SNN

The key is that k-NN **augments** SNN input, not replaces it:

```python
# SNN input structure (608 neurons):
snn_input = concat([
    vlm_input,              # 64: High-level goals
    milestone_input,        # 64: Story flags
    quest_input,            # 64: Quest objectives
    cnn_input,              # 64: Local vision
    map_input,              # 64: Global position
    spatial_nav_input,      # 64: Walkability
    slam_planning_input,    # 64: Path planning
    milestone_feedback,     # 64: Server progress
    battle_strategy,        # 64: Event spikes + strategy
    episodic_recall        # 32: k-NN augmentation (NEW!)
])
```

**Fusion Formula:**
```python
augmented_context = concat(
    snn_embed,                    # Learned temporal state
    knn_action_suggestion,        # Episodic recall
    knn_reward_estimate,
    knn_milestone_likelihood
)
→ policy_input
```

This creates:
- **SNN** = "How current step fits in long temporal sequence"
- **k-NN** = "What worked last time in similar situations"

## Battle Event Spikes

To enable temporal credit assignment, battle outcomes are encoded as event spikes:

```python
battle_events = {
    'battle_started': bool,
    'battle_won': bool,
    'battle_lost': bool,
    'pokemon_fainted': bool,
    'gym_leader_battle': bool,
    'gym_leader_won': bool,
    'used_item': bool,
    'super_effective_hit': bool
}
```

**SNN Input Format (64-dim):**
```
[0-7]:  Event spikes (8-dim binary)
[8-63]: Strategy knowledge (56-dim continuous)
```

**This enables SNN learning:**
```
Pattern: "Low safety warning" + ignore → battle_lost spike
         → SNN learns to trust safety warnings

Pattern: Water move + type advantage → super_effective + battle_won
         → SNN reinforces type effectiveness
```

## Implementation Details

### EpisodicMemory Class

```python
class EpisodicMemory:
    def __init__(self, max_size=10000, k=5):
        self.nav_memory = deque(maxlen=max_size)
        self.battle_memory = deque(maxlen=max_size)
        self.quest_memory = deque(maxlen=max_size)
        self.k = k
    
    def store_navigation_experience(self, cnn_features, story_vec, 
                                    waypoint_hint, action, reward, milestone):
        """Store compact experience key for later recall"""
        
    def query_navigation(self, current_features):
        """Retrieve k nearest neighbors by L2 distance"""
        # Returns: suggested_actions, action_weights, avg_reward, milestone_likelihood
    
    def get_episodic_augmentation(self, domain, query_features):
        """Main interface: returns 32-dim vector for SNN input"""
```

### BattleExperienceTracker Enhancement

```python
class BattleExperienceTracker:
    def __init__(self):
        self.current_battle_events = {
            'battle_started': False,
            'battle_won': False,
            'battle_lost': False,
            # ... 8 total events
        }
    
    def fire_event(self, event_name):
        """Trigger event spike for SNN temporal learning"""
        self.current_battle_events[event_name] = True
    
    def get_battle_event_spikes(self):
        """Return 8-dim binary vector of current events"""
        return [float(event) for event in self.current_battle_events.values()]
    
    def reset_event_spikes(self):
        """Clear spikes after SNN reads them"""
    
    def encode_for_snn(self):
        """Returns [event_spikes(8), strategy_knowledge(56)] = 64-dim"""
```

## What NOT to Do with k-NN

❌ **Do NOT use k-NN to select actions directly**
- This leads to degeneracy (agent repeats past actions without learning)

❌ **Do NOT replace RL with k-NN**
- RL is needed for long-horizon reasoning and exploration

❌ **Do NOT replace SNN with k-NN**
- SNN handles real temporal dependencies (credit assignment, context)
- k-NN only does similarity search

❌ **Do NOT store raw frames**
- Always embed first (CNN → compact features)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ENVIRONMENT (Pokemon Emerald)             │
└───────────────────────┬─────────────────────────────────────┘
                        │ state
                        ▼
          ┌─────────────────────────────┐
          │   CNN (Local Vision)        │
          │   - 84x84x3 frame          │
          │   - Visual features         │
          └──────────┬──────────────────┘
                     │ features
                     ▼
    ┌────────────────────────────────────────┐
    │        k-NN EPISODIC MEMORY             │
    │  ┌──────────────────────────────────┐  │
    │  │  Navigation Memory (deque)       │  │
    │  │  • cnn_feat + story + waypoint   │  │
    │  │  → action, reward, milestone     │  │
    │  └──────────────────────────────────┘  │
    │  ┌──────────────────────────────────┐  │
    │  │  Battle Memory (deque)           │  │
    │  │  • battle_vec + hp + type_adv    │  │
    │  │  → action, outcome, gym_flag     │  │
    │  └──────────────────────────────────┘  │
    │                                         │
    │  query_navigation() → 32-dim vector    │
    └────────────┬────────────────────────────┘
                 │ episodic_recall (32-dim)
                 ▼
    ┌────────────────────────────────────────┐
    │           SNN CORE (608 neurons)        │
    │  ┌──────────────────────────────────┐  │
    │  │  Input Streams:                  │  │
    │  │  1. VLM goals          (64)      │  │
    │  │  2. Milestones         (64)      │  │
    │  │  3. Quest objectives   (64)      │  │
    │  │  4. CNN vision         (64)      │  │
    │  │  5. Map position       (64)      │  │
    │  │  6. Spatial nav        (64)      │  │
    │  │  7. SLAM planning      (64)      │  │
    │  │  8. Progress feedback  (64)      │  │
    │  │  9. Battle events      (64)      │  │
    │  │     ├─ spikes(8)                 │  │
    │  │     └─ strategy(56)              │  │
    │  │  10. Episodic recall   (32) ⭐   │  │
    │  └──────────────────────────────────┘  │
    │                                         │
    │  LIF Dynamics:                          │
    │  V(t+1) = decay * V(t) + I(t)          │
    │  spike = V > threshold                  │
    │                                         │
    │  Temporal Integration:                  │
    │  • Learned patterns (SNN weights)      │
    │  • Episodic recall (k-NN memory)       │
    └────────────┬────────────────────────────┘
                 │ snn_embed (608-dim)
                 ▼
    ┌────────────────────────────────────────┐
    │      FUSION: SNN + CNN + k-NN          │
    │                                         │
    │  augmented_context = concat(           │
    │     snn_embed,                          │
    │     cnn_local_vision,                   │
    │     knn_suggestions                     │
    │  )                                      │
    └────────────┬────────────────────────────┘
                 │
                 ▼
    ┌────────────────────────────────────────┐
    │          RL POLICY (PPO)                │
    │                                         │
    │  Actor: action_probs = softmax(π(s))   │
    │  Critic: value = V(s)                   │
    └────────────┬────────────────────────────┘
                 │ action
                 ▼
    ┌────────────────────────────────────────┐
    │         STORE EXPERIENCE                │
    │                                         │
    │  episodic_memory.store(                 │
    │     cnn_feat, story, waypoint,          │
    │     action, reward, milestone           │
    │  )                                      │
    │                                         │
    │  battle_tracker.fire_event(             │
    │     'battle_won' / 'battle_lost'        │
    │  )                                      │
    └─────────────────────────────────────────┘
```

## Summary: Why This Works

1. **Complementary Systems:**
   - SNN learns temporal patterns through backpropagation
   - k-NN retrieves similar past experiences without training

2. **Episodic Augmentation:**
   - k-NN provides "hints" based on similarity
   - SNN integrates these hints with learned temporal reasoning
   - Policy makes final decision using both

3. **Battle Credit Assignment:**
   - Event spikes (battle_won, battle_lost) provide temporal markers
   - SNN learns which patterns lead to success/failure
   - k-NN recalls what worked in similar matchups

4. **No Redundancy:**
   - k-NN is NOT doing novelty detection (that's redundant)
   - k-NN is doing episodic recall (unique contribution)
   - Each component has a distinct purpose

## Expected Improvements

1. **Faster Learning in Similar Situations:**
   - Agent recalls successful actions from past experiences
   - Reduces exploration in known-good scenarios

2. **Better Battle Strategy:**
   - Remembers type matchups that worked
   - Avoids repeating failed strategies

3. **Improved Navigation:**
   - Recalls successful paths through complex areas
   - Handles visual ambiguity (similar-looking locations)

4. **Stuck Recovery:**
   - Detects repetitive failure patterns
   - Suggests alternative actions from memory

## Result: Hybrid Agent

```
VLM + SLAM + SNN + CNN + k-NN Episodic Memory Agent
```

A truly hybrid system that combines:
- **Semantic reasoning** (VLM)
- **Geometric mapping** (SLAM)
- **Temporal learning** (SNN)
- **Spatial perception** (CNN)
- **Episodic memory** (k-NN)

For long-horizon game speedrunning with both learning and recall.
