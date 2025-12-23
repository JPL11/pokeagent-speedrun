# Answers to Your Questions

## 1. What is NOOP?

**NOOP = "No Operation"** - means "do nothing" for one game frame.

Your agent has **7 actions**:
- `A`, `B`, `UP`, `DOWN`, `LEFT`, `RIGHT`, **`NOOP`**

### Why NOOP is Important:
- **Wait for dialogue** to finish before pressing A again
- **Let NPCs move** without interfering  
- **Observe before acting** - strategic pauses
- **Prevent button spam** that can softlock menus

Without NOOP, the agent constantly spams buttons which can:
- Skip important dialogue too fast
- Get stuck in menu loops
- Miss timing-dependent events

---

## 2. Current Milestones (11 Total)

```
✅ 1. TITLE_SCREEN       - Press A to start game
✅ 2. CHARACTER_CREATED  - Complete character creation
🎯 3. LEFT_HOUSE         - Exit starter house to Littleroot Town  
   4. ROUTE_101          - Reach Route 101 (save Prof Birch)
   5. GOT_STARTER        - Receive first Pokemon from Birch
   6. VISITED_LAB        - Visit Birch's lab for Pokedex
   7. ROUTE_103          - Travel to Route 103, battle rival
   8. OLDALE_TOWN        - First new town (+50 reward!)
   9. PETALBURG_CITY     - Meet Norman (your dad) at gym
   10. RUSTBORO_CITY     - Reach Rustboro for first gym
   11. FIRST_GYM_BADGE   - Defeat Roxanne (Stone Badge)
```

**Reward Structure:**
- 🎯 **+100**: Milestone completion (HUGE!)
- 🏙️ **+50**: New town/city
- 🛣️ **+30**: New route
- 🏠 **+20**: Building entered
- 🗺️ **+5**: New tile explored
- ❌ **-0.5**: Revisiting same spot >10 times

---

## 3. Why High CPU but Low GPU Usage?

**Your Current Stats:**
```
GPU: 0% utilization
VRAM: 809 MB used (out of 16GB)
CPU: High utilization
Speed: ~2 steps/second
```

### Root Causes:

#### A. **VLM Only Runs Every 100 Steps**
- Qwen 2B VLM loads at startup (809MB VRAM)
- Only makes inference every 100 steps
- GPU idle 99% of the time waiting for VLM calls

#### B. **PPO Batches Every 64 Steps**
- Training happens in bursts every 64 steps
- GPU processes batch quickly, then idles
- Most time spent collecting experience

#### C. **CPU Bottlenecks:**
1. **Game Server** - mGBA emulation (CPU-based)
2. **HTTP Requests** - Python requests library overhead
3. **Image Processing** - Base64 decode → PIL → numpy (CPU)
4. **SNN Encoding** - Python loops with LIF neurons (CPU)
5. **Frame Cache Writing** - JSON serialization

#### D. **Sequential Processing:**
```
Step → HTTP request → Wait for server → Decode image → 
SNN encode → Store frame → Send action → Repeat
```

### Solutions:

**To increase GPU usage:**
1. Run multiple parallel environments (4-8 copies)
2. Batch VLM calls every 50 steps instead of 100
3. Move SNN to GPU with PyTorch
4. Use native mGBA bindings (no HTTP)

**Current bottleneck:** Game emulation speed (~2 FPS), not training

---

## 4. PokemonRedExperiments Event Flags System

### What Are Event Flags?

**Event flags** = Binary flags in game memory tracking story progress

Example from Pokemon Red (memory addresses 0xD747-0xD87E):
```
Bit 0x01: Talked to Oak
Bit 0x02: Got Pokedex
Bit 0x05: Defeated Brock
Bit 0x12: Entered Cerulean City
Bit 0x23: S.S. Anne departed
... ~300+ flags total
```

### PokemonRedExperiments Reward Structure:

```python
state_scores = {
    "event": update_max_event_rew() * 4,      # 4X WEIGHT!
    "badge": get_badges() * 10,
    "heal": total_healing * 10,
    "explore": len(seen_coords) * 0.1,
    "stuck": coord_revisits * -0.05
}
```

**Key insight:** Event flags get **4x weight** - ensures story progression!

### How They Track Events:

```python
# Check all event flag bytes
for address in range(event_flags_start, event_flags_end):
    val = read_memory(address)
    for bit in range(8):
        if (val >> bit) & 1:
            # This event flag is set!
            current_events.add(f"0x{address:X}-{bit}")

# Reward for NEW events
new_events = current_events - previous_events
reward += len(new_events) * 4
```

### Applying to Pokemon Emerald:

**We need to:**
1. Find Emerald's event flag memory addresses
2. Track which flags represent important milestones:
   - Talked to Birch
   - Got Pokedex
   - Entered Route 101
   - Defeated gym leaders
   - Entered Pokemon Centers
   - Talked to important NPCs

3. Give bonus rewards for NEW flags

**Example for Emerald:**
```python
# Emerald event flags (need to find addresses)
EMERALD_EVENT_FLAGS = {
    "RESCUED_BIRCH": 0x????-bit?,
    "GOT_POKEDEX": 0x????-bit?,
    "HEALED_POKEMON": 0x????-bit?,
    "GYM_LEADER_1": 0x????-bit?,
}

# Track and reward
reward += len(new_event_flags) * 5
```

### Why This Helps:

Without event tracking, agent just wanders exploring tiles. With event flags:
- **Enters buildings** (sets event flags)
- **Talks to NPCs** (sets event flags)  
- **Progresses story** (sets event flags)
- Gets HUGE rewards for actual game progress!

---

## Next Steps to Improve:

1. **Add Emerald Event Flag Tracking** (like PokemonRedExperiments)
2. **Increase GPU utilization** with parallel environments
3. **Reduce VLM frequency** from 100 to 50 steps
4. **Track building entries** more explicitly with event flags
5. **Add NPC dialogue detection** with rewards

This will ensure the agent actually progresses through the game story instead of just exploring randomly!
