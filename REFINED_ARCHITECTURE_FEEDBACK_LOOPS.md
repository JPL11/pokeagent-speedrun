# REFINED ARCHITECTURE - Battle Training with Feedback Loops

## Problem with Current Design

**Current approach**: Battle engagement strategy is **procedural** (if-else rules)
**Issue**: Doesn't integrate with VLM high-level reasoning or SNN temporal learning

## Optimal Architecture - Three-Tier System

```
┌─────────────────────────────────────────────────────────┐
│  TIER 1: VLM HIGH-LEVEL REASONING (Strategic)          │
│  "Should I grind now or proceed to gym?"                │
│  Input: Current level, target level, milestone priority │
│  Output: Strategic directive to SNN                     │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 2: SNN TEMPORAL REASONING (Tactical)              │
│  "How many more battles until target?"                  │
│  Input: Battle history, level gains, HP state           │
│  Output: Action bias (fight vs escape, use item vs save)│
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  TIER 3: REWARD SHAPING (Feedback)                      │
│  "Reinforce successful patterns"                        │
│  Input: Actions taken, outcomes achieved                │
│  Output: Reward signals to train SNN                    │
└─────────────────────────────────────────────────────────┘
```

## Tier 1: VLM Strategic Reasoning (High-Level)

### What VLM Should Decide:
- **"Do I need to train?"** → Check party level vs next milestone requirement
- **"Where should I go?"** → Gym if strong enough, grinding area if weak
- **"What's my current goal?"** → "Level to 13 for Roxanne" or "Enter gym now"

### VLM Prompt Enhancement:

```python
# In ServerAPIOracle._build_vlm_prompt_with_memory()

# Calculate strategic assessment
party_levels = [p.get('level', 0) for p in party if p]
avg_level = sum(party_levels) / max(len(party_levels), 1)
next_milestone = self._get_next_milestone_name(milestones_completed)
required_level = MILESTONE_LEVEL_REQUIREMENTS.get(next_milestone, 10)

level_status = ""
if avg_level < required_level - 3:
    level_status = "⚠️ UNDER-LEVELED - TRAINING REQUIRED"
    strategic_directive = "GRIND"
elif avg_level < required_level:
    level_status = "🟡 SLIGHTLY WEAK - TRAIN OR PROCEED"
    strategic_directive = "OPTIONAL_GRIND"
else:
    level_status = "✅ STRONG ENOUGH - PROCEED TO MILESTONE"
    strategic_directive = "PROCEED"

# Add to VLM prompt
strategic_context = f"""
📊 STRATEGIC ASSESSMENT
─────────────────────────────────────────────────────────
Current Level: {avg_level:.1f}
Required Level: {required_level} (for {next_milestone})
Status: {level_status}

STRATEGIC DIRECTIVE: {strategic_directive}

{'→ Find TALL GRASS area (Route 101/102/103)' if strategic_directive == 'GRIND' else ''}
{'→ Fight wild Pokemon to gain EXP and level up' if strategic_directive == 'GRIND' else ''}
{'→ Target: Reach level {required_level} before attempting {next_milestone}' if strategic_directive == 'GRIND' else ''}
{'→ Continue to next story objective' if strategic_directive == 'PROCEED' else ''}
{'→ Note: Unavoidable trainer battles will occur en route' if strategic_directive == 'PROCEED' else ''}
"""
```

**Key Insight**: VLM makes **goal decision** ("I need to grind" vs "I'm ready for gym"), not micro-decisions ("fight this specific wild Pokemon").

## Tier 2: SNN Temporal Reasoning (Tactical)

### What SNN Should Learn:
- **Battle frequency patterns**: "When grinding, fight ~80% of encounters"
- **HP management**: "Used potion at 40% HP → survived → good pattern"
- **Level progression**: "Fought 10 battles → gained 1 level → need 10 more"
- **Grinding duration**: "Grinding for 500 steps → reached target → exit mode"

### SNN Input Augmentation:

```python
# NEW INPUT 12: Strategic Context Vector (32-dim)
# Encodes VLM's high-level strategic state

def encode_strategic_context(strategic_directive, party_state, battle_state):
    """
    Encode strategic context for SNN temporal reasoning.
    
    Returns 32-dim vector:
    [0-7]: Strategic mode (one-hot)
           0: PROCEED_TO_MILESTONE
           1: GRIND_REQUIRED
           2: GRIND_OPTIONAL
           3: HEAL_REQUIRED
           4: SHOP_REQUIRED
           5: PREPARE_FOR_GYM
           6: IN_GYM_BATTLE
           7: POST_BLACKOUT
    
    [8-15]: Party state
           8: Avg HP ratio (0-1)
           9: Avg level / 100
           10: Level deficit vs target / 10
           11: Has healing items? (0/1)
           12: Number of fainted Pokemon / 6
           13-15: Reserved
    
    [16-23]: Battle context
           16: In battle? (0/1)
           17: Opponent level / 100
           18: Grinding mode active? (0/1)
           19: Steps since last level up / 1000
           20: Battles fought this session / 50
           21: Wild battle win rate (0-1)
           22-23: Reserved
    
    [24-31]: Temporal feedback
           24: Steps in grinding mode / 1000
           25: Recent reward trend (moving avg)
           26: Milestone progress ratio (0-1)
           27: Distance to grinding area (if grinding mode)
           28-31: Reserved
    """
    
    strategic_vec = np.zeros(32, dtype=np.float32)
    
    # Strategic mode (one-hot)
    mode_map = {
        'PROCEED': 0,
        'GRIND': 1,
        'OPTIONAL_GRIND': 2,
        'HEAL': 3,
        'SHOP': 4,
        'PREPARE_GYM': 5,
        'GYM_BATTLE': 6,
        'POST_BLACKOUT': 7
    }
    mode_idx = mode_map.get(strategic_directive, 0)
    strategic_vec[mode_idx] = 1.0
    
    # Party state
    party_levels = [p.get('level', 0) for p in party_state if p]
    avg_level = sum(party_levels) / max(len(party_levels), 1)
    
    party_hp_ratios = [
        p.get('current_hp', 0) / max(p.get('max_hp', 1), 1)
        for p in party_state if p
    ]
    avg_hp = sum(party_hp_ratios) / max(len(party_hp_ratios), 1)
    
    strategic_vec[8] = avg_hp
    strategic_vec[9] = avg_level / 100.0
    strategic_vec[10] = max(0, required_level - avg_level) / 10.0
    strategic_vec[11] = 1.0 if has_healing_items else 0.0
    strategic_vec[12] = sum(1 for p in party_state if p.get('current_hp', 0) == 0) / 6.0
    
    # Battle context
    strategic_vec[16] = 1.0 if battle_state.get('in_battle') else 0.0
    strategic_vec[17] = battle_state.get('opponent_level', 0) / 100.0
    strategic_vec[18] = 1.0 if grinding_mode_active else 0.0
    strategic_vec[19] = steps_since_level_up / 1000.0
    strategic_vec[20] = battles_this_session / 50.0
    strategic_vec[21] = wild_battle_win_rate
    
    # Temporal feedback
    strategic_vec[24] = grinding_steps_elapsed / 1000.0
    strategic_vec[25] = recent_reward_moving_avg / 10.0
    strategic_vec[26] = milestones_completed / total_milestones
    
    return strategic_vec
```

**Key Insight**: SNN receives **strategic context** from VLM as input, learns **temporal patterns** of how to execute that strategy.

## Tier 3: Feedback-Based Reward Shaping

### Dynamic Reward Modulation Based on Strategic Mode

```python
# In ServerAwareEnv.step() reward calculation

# Get current strategic directive from VLM
strategic_directive = getattr(self, 'current_strategic_directive', 'PROCEED')

# ========== ADAPTIVE REWARD SHAPING ==========

# 1. LEVEL UP REWARD (context-dependent)
if level_gain > 0:
    if strategic_directive == 'GRIND':
        # In grinding mode - HIGH reward for leveling
        level_reward = 8.0 * level_gain
        info_parts.append(f"📈 LEVEL UP (grinding) +{level_reward:.1f}")
    elif strategic_directive == 'PROCEED':
        # Not grinding - moderate reward (still good, but not focus)
        level_reward = 3.0 * level_gain
        info_parts.append(f"📈 Level up +{level_reward:.1f}")
    else:
        level_reward = 5.0 * level_gain
    
    reward += level_reward

# 2. BATTLE WIN REWARD (context-dependent)
if battle_won:
    if strategic_directive == 'GRIND':
        # In grinding mode - reward each battle
        battle_reward = 4.0
        info_parts.append(f"⚔️ Training battle +{battle_reward:.1f}")
    elif strategic_directive == 'PROCEED':
        # Not grinding - small reward (prefer avoiding unnecessary battles)
        battle_reward = 1.0
        info_parts.append(f"⚔️ Battle +{battle_reward:.1f}")
    else:
        battle_reward = 2.0
    
    reward += battle_reward

# 3. GRASS TRAVERSAL REWARD (context-dependent)
if in_tall_grass:
    if strategic_directive == 'GRIND':
        # Looking for battles - REWARD being in grass
        grass_reward = 2.5
        info_parts.append(f"🌿 Grinding area +{grass_reward:.1f}")
    elif strategic_directive == 'PROCEED':
        # Not grinding - neutral (need to cross grass anyway)
        grass_reward = 0.2
    else:
        grass_reward = 1.0
    
    reward += grass_reward

# 4. STRATEGIC GOAL COMPLETION (HIGH reward)
# Check if strategic goal achieved this step
if strategic_directive == 'GRIND':
    # Check if reached target level
    if avg_level >= required_level:
        goal_completion_reward = 50.0  # BIG reward
        info_parts.append(f"🎯 TRAINING COMPLETE! +{goal_completion_reward:.1f}")
        reward += goal_completion_reward
        
        # Update strategic directive (feedback loop)
        self.current_strategic_directive = 'PROCEED'
        
        # Fire SNN spike for goal completion
        self.battle_tracker.fire_event('training_goal_achieved')

elif strategic_directive == 'HEAL':
    # Check if HP restored
    if avg_hp >= 0.8:
        heal_completion_reward = 25.0
        info_parts.append(f"❤️ HEALED UP! +{heal_completion_reward:.1f}")
        reward += heal_completion_reward
        
        # Can now proceed or grind
        self.current_strategic_directive = 'PROCEED'

# 5. DIALOG ANTI-FARMING PENALTY (Nov 23, 2024) ★★★ NEW
# Prevents exploitation of constant dialog rewards
if game_state == 'dialog':
    if not hasattr(self, 'consecutive_dialog_steps'):
        self.consecutive_dialog_steps = 0
    self.consecutive_dialog_steps += 1
    
    if action_name in ['A', 'B']:
        if self.consecutive_dialog_steps > 5:
            # Heavy penalty for dialog farming!
            dialog_penalty = -2.0
            reward += dialog_penalty
            info_parts.append(f"❌ Dialog farming penalty {dialog_penalty:.1f}")
        else:
            # Minimal reward for normal dialog
            dialog_reward = 0.1  # Reduced from 0.5
            reward += dialog_reward
            info_parts.append(f"💬 Dialog {action_name} +{dialog_reward:.1f}")
else:
    # Reset counter when not in dialog
    if hasattr(self, 'consecutive_dialog_steps'):
        self.consecutive_dialog_steps = 0

# 6. MILESTONE PROGRESS (HIGHEST priority - unchanged)
if new_milestone_achieved:
    milestone_reward = 500.0  # DOMINANT reward
    reward += milestone_reward
    info_parts.append(f"⭐ MILESTONE! +{milestone_reward:.1f}")
```

### Updated Reward Hierarchy (Nov 23, 2024)

```
Milestone (+500)             ← Highest: Story progress
  └─ Training Complete (+50)     ← Strategic goals
      └─ Healed Up (+25)             ← Resource management
          └─ VLM Guidance (+8.0)         ← Tactical alignment
              └─ Level Up (8.0× GRIND)       ← Context-aware
                  └─ Battle Win (4.0 GRIND)      ← Context-aware
                      └─ Grass (2.5 GRIND)           ← Context-aware
                          └─ Exploration (+3.2)          ← Discovery
                              └─ Dialog (+0.1)               ← Minimal
                                  └─ Dialog Farming (-2.0)       ← PENALTY
```

**Key Insight**: Dialog farming prevention enforces reward hierarchy by making constant-reward exploitation unprofitable. Agent must follow strategic guidance (VLM +8.0) rather than spamming NPCs (dialog +0.1, or -2.0 if farming).

## Feedback Loop Mechanism

### Strategic Directive Updates (Dynamic Goal Adaptation)

```python
class StrategicDirectiveManager:
    """
    Manages high-level strategic directives and feedback loops.
    Updates strategy based on agent's progress.
    """
    
    def __init__(self):
        self.current_directive = 'PROCEED'
        self.directive_history = []
        
        # Feedback tracking
        self.grinding_session_start_level = 0
        self.grinding_session_start_step = 0
        self.levels_gained_this_session = 0
    
    def update_directive(self, party_state, milestones, location):
        """
        Update strategic directive based on current state.
        Called every VLM invocation (~25 steps).
        """
        party_levels = [p.get('level', 0) for p in party_state if p]
        avg_level = sum(party_levels) / max(len(party_levels), 1)
        
        party_hp = [
            p.get('current_hp', 0) / max(p.get('max_hp', 1), 1)
            for p in party_state if p
        ]
        avg_hp = sum(party_hp) / max(len(party_hp), 1)
        
        next_milestone = self._get_next_milestone(milestones)
        required_level = MILESTONE_REQUIREMENTS[next_milestone]['level']
        
        # PRIORITY 1: Critical HP - MUST heal
        if avg_hp < 0.3:
            self._transition_to('HEAL', reason='CRITICAL_HP')
            return
        
        # PRIORITY 2: Post-blackout - recover before proceeding
        if hasattr(self, 'just_blackouted') and self.just_blackouted:
            self._transition_to('POST_BLACKOUT', reason='RECOVERY')
            self.just_blackouted = False
            return
        
        # PRIORITY 3: Approaching gym - prepare
        if self._is_near_gym(location, next_milestone):
            if avg_level < required_level - 1:
                self._transition_to('PREPARE_GYM', reason='UNDER_LEVELED_FOR_GYM')
            elif avg_hp < 0.6:
                self._transition_to('HEAL', reason='PREPARE_FOR_GYM')
            else:
                self._transition_to('PROCEED', reason='READY_FOR_GYM')
            return
        
        # PRIORITY 4: Level check for next milestone
        level_deficit = required_level - avg_level
        
        if level_deficit >= 3:
            # Significantly under-leveled - MUST grind
            self._transition_to('GRIND', reason=f'NEED_{level_deficit}_LEVELS')
            
        elif level_deficit >= 1:
            # Slightly under-leveled - optional grinding
            if self.current_directive != 'GRIND':
                # Don't start grinding for small deficit
                self._transition_to('OPTIONAL_GRIND', reason='SLIGHT_DEFICIT')
            # If already grinding, continue
        
        else:
            # Strong enough - proceed to milestone
            if self.current_directive == 'GRIND':
                # Grinding complete!
                self._transition_to('PROCEED', reason='TRAINING_COMPLETE')
            else:
                self._transition_to('PROCEED', reason='STRONG_ENOUGH')
        
        # FEEDBACK: Check grinding progress
        if self.current_directive == 'GRIND':
            self._check_grinding_progress(avg_level)
    
    def _transition_to(self, new_directive, reason):
        """Transition to new strategic directive with logging."""
        if new_directive != self.current_directive:
            old_directive = self.current_directive
            self.current_directive = new_directive
            
            self.directive_history.append({
                'from': old_directive,
                'to': new_directive,
                'reason': reason,
                'step': self.step_count
            })
            
            logger.info(f"🎯 STRATEGIC CHANGE: {old_directive} → {new_directive} ({reason})")
            
            # Initialize new directive
            if new_directive == 'GRIND':
                self.grinding_session_start_level = self._get_avg_level()
                self.grinding_session_start_step = self.step_count
                self.levels_gained_this_session = 0
    
    def _check_grinding_progress(self, current_level):
        """
        Monitor grinding progress and adjust if needed.
        Feedback loop: If grinding not working, change strategy.
        """
        steps_elapsed = self.step_count - self.grinding_session_start_step
        levels_gained = current_level - self.grinding_session_start_level
        
        # Feedback: Not making progress?
        if steps_elapsed > 1000 and levels_gained < 1:
            logger.warning(f"⚠️ GRINDING INEFFECTIVE: {steps_elapsed} steps, {levels_gained} levels")
            logger.warning(f"   → Consider: Wrong location, avoiding battles, or need evolution")
            
            # Transition to PROCEED (might need to find better grinding spot)
            self._transition_to('PROCEED', reason='GRINDING_INEFFECTIVE')
        
        # Feedback: Making good progress?
        elif levels_gained >= 1:
            self.levels_gained_this_session = levels_gained
            logger.info(f"✅ GRINDING PROGRESS: {levels_gained} level(s) in {steps_elapsed} steps")
```

## Integration Summary

### **High-Level (VLM)** - Strategic Goal Setting:
- "I need to reach level 13 for Rustboro Gym"
- "Current level 8 → 5 levels needed → GRIND directive"
- Encodes directive into 32-dim strategic context vector

### **Mid-Level (SNN)** - Temporal Execution:
- Receives strategic context as INPUT 12 (32-dim)
- Learns: "When GRIND directive active → fight battles → gain levels"
- Learns: "When PROCEED directive → avoid unnecessary battles → reach milestone"
- Temporal credit assignment: "Fought 10 battles → leveled up → good pattern"

### **Low-Level (Rewards)** - Feedback Shaping:
- Dynamic rewards based on strategic directive:
  - GRIND mode: High reward for battles/levels (+8.0)
  - PROCEED mode: High reward for milestones (+500), low for battles (+1.0)
- Goal completion reward: +50.0 when training target reached
- Feedback loop: Target reached → update directive → change reward structure

## Comparison: Old vs New

### Old Design (Procedural):
```python
if avg_level < target_level:
    if random.random() < 0.6:
        action = 'FIGHT'  # Hardcoded probability
```

### New Design (Learned):
```python
# VLM sets strategic directive
strategic_directive = 'GRIND'  # High-level goal

# SNN receives strategic context + learns patterns
strategic_vec = encode_strategic_context(directive, party, battle)
snn_input = [..., strategic_vec]  # INPUT 12

# SNN output influences policy
# Agent learns: "When directive=GRIND + in_grass → FIGHT action → +8 reward → repeat"
```

## Why This Is Better

1. **Separation of Concerns**:
   - VLM: Strategic decisions (what to do)
   - SNN: Temporal patterns (how to do it)
   - Rewards: Feedback signals (reinforce success)

2. **Learned Behavior**:
   - SNN learns optimal grinding behavior (not hardcoded rules)
   - Adapts to different situations through experience
   - Temporal credit assignment: "This sequence of actions led to level up"

3. **Feedback Loops**:
   - Target reached → directive updates → rewards adjust
   - Ineffective grinding → change location
   - Strong enough → stop grinding → proceed to milestone

4. **Unavoidable Battles Handled**:
   - PROCEED directive: Low reward for wild battles, high for progress
   - Agent learns: "Skip wild Pokemon, but trainers are mandatory"
   - Healing strategy: Heal before entering trainer-dense areas

## Implementation Status

✅ **NOT APPLIED YET** - Waiting for current training to complete

**Advantages**:
- Cleaner architecture (VLM/SNN/Reward separation)
- More flexible (learns patterns vs hardcoded rules)
- Better feedback loops (dynamic directive updates)
- Handles "strong enough" → "proceed" transition naturally

**Your instinct was correct**: Battle strategy should be **high-level reasoning** (VLM), grinding duration should be **temporal** (SNN), with **feedback loops** connecting them.

This refined design is ready to implement after your 2M training completes!
