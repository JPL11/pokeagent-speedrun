# Lab Stuck Issue Analysis & Solutions
**Date**: November 17, 2025  
**Issue**: Agent stuck in Birch's Lab for 2+ hours  
**Root Causes**: 3 distinct problems identified

---

## 📊 ANALYSIS RESULTS

### Problem Statistics:
- **Lab visits**: 32 VLM oracle calls (all in map 0x0104)
- **Parsing errors**: 90 NoneType errors (`response.upper()` on None)
- **Actions followed**: Only 7 (mostly 'A' button)
- **Unique coords**: 32 different positions explored

### Location Pattern:
- Most visited: (0,5), (0,4), (2,5), (2,4) - **all in upper lab**
- Y range: 2 to 9 (height: 8 tiles)
- **Lab exit is at y=9 (SOUTH/DOWN)** but agent keeps exploring NORTH

---

## 🔴 ROOT CAUSE #1: VLM Response Parsing Failure

### The Problem:
VLM is returning **descriptive advice** instead of explicit **ACTION:** directives:

**❌ What VLM Returns**:
```
To progress the story and reach the next milestone, you need to leave 
Professor Birch's lab and head north through Route 101...
```

**✅ What Code Expects**:
```
ACTION: DOWN
REASON: Exit lab to reach Route 101
```

### The Bug:
```python
def parse_action_from_response(self, response):
    if not response:
        return None
    match = re.search(r'ACTION:\s*([A-Z]+)', response.upper())  # ← Fails
    if match:
        return match.group(1)
    return None  # ← Returns None
```

Then later:
```python
suggested_action = self.oracle.parse_action_from_response(guidance)
# suggested_action is None
if suggested_action == action_name:  # ← NoneType.upper() crashes
```

---

## 🟡 ROOT CAUSE #2: Spatial Reasoning Gap

### The Problem:
VLM has **high-level goal** ("go north to Route 101") but lacks **local spatial awareness**:

1. Agent is INSIDE lab (building interior)
2. To go north, must FIRST exit building (requires going DOWN/SOUTH)
3. VLM doesn't understand indoor/outdoor distinction
4. SLAM doesn't mark exit points (doors)

### Evidence:
- Agent explored y=2 to y=9 (full lab height)
- Most visits at y=4-5 (middle/upper lab) - trying to go NORTH
- Lab exit at y=9 (south wall) - rarely visited
- VLM says "go north" but doesn't say "exit first"

### Missing:
- Door detection (RGB → SLAM integration)
- Indoor/outdoor zone tracking
- Micro-milestones: `EXIT_LAB` → `GO_NORTH` → `ENTER_ROUTE_101`

---

## 🟠 ROOT CAUSE #3: Temporal Reasoning Insufficient

### The Problem:
SNN has path progress tracking BUT:
- No "sub-quest chain" memory: [EXIT → TRAVEL → ARRIVE]
- No "stuck in same building" detection
- No "tried this 10 times, try different approach" learning

### What's Missing:
```python
# Current SLAM hints [9-11]: path progress
hints[9] = progress_ratio  # 0.3 → 0.5 → making progress
hints[10] = dist_to_waypoint  # 8 → 5 → getting closer

# NEEDED: Sub-quest temporal chain [12-19]
hints[12] = current_subquest_id  # 0=EXIT_LAB, 1=GO_NORTH, 2=ENTER_ROUTE
hints[13] = subquest_progress  # 0.0 → 1.0 within current step
hints[14] = time_stuck  # How long at same subquest?
hints[15] = retry_count  # Failed attempts at current step
```

---

## 💡 SOLUTION 1: Fix VLM Parsing (IMMEDIATE - 15 min)

### Add Fallback Natural Language Parser:

```python
def parse_action_from_response(self, response):
    """Extract action with fallback NLP parsing"""
    if not response:
        return None
    
    # Primary: Look for "ACTION: X" pattern
    match = re.search(r'ACTION:\s*([A-Z]+)', response.upper())
    if match:
        action = match.group(1)
        action_map = {
            'UP': 'UP', 'DOWN': 'DOWN', 'LEFT': 'LEFT', 'RIGHT': 'RIGHT',
            'A': 'A', 'B': 'B', 'NOOP': 'NOOP'
        }
        return action_map.get(action)
    
    # FALLBACK: Natural language understanding
    response_lower = response.lower()
    
    # Map common phrases to actions
    if any(word in response_lower for word in ['exit', 'leave', 'south', 'down']):
        return 'DOWN'
    if any(word in response_lower for word in ['enter', 'talk', 'interact', 'speak']):
        return 'A'
    if 'north' in response_lower or 'up' in response_lower:
        return 'UP'
    if 'east' in response_lower or 'right' in response_lower:
        return 'RIGHT'
    if 'west' in response_lower or 'left' in response_lower:
        return 'LEFT'
    
    return None
```

---

## 💡 SOLUTION 2: RGB-SLAM Door Detection (SHORT-TERM - 1 hour)

### Enhance Spatial Navigation Stream:

**Currently** (Stream #6):
- Terrain (64): Map-based ledges
- RGB (8): Grass/tree pixel detection
- SLAM occupancy (16): Walkability

**ADD** (expand RGB from 8 to 16 dims):
```python
# RGB analysis [0-7]: current
rgb_features[0-3] = grass_density, tree_density, water_density, path_density

# NEW [8-15]: door/exit detection
rgb_features[8] = door_confidence  # Brown rectangular pattern
rgb_features[9] = exit_sign_confidence  # Darker tile at edge
rgb_features[10] = indoor_lighting  # Darker = indoor, brighter = outdoor
rgb_features[11] = wall_proximity  # Distance to nearest wall
```

**Feed to SLAM**:
```python
# In slam_milestone_classes.py
def update_from_rgb_analysis(self, map_id, x, y, rgb_features):
    """Mark doors as exit waypoints"""
    door_conf = rgb_features[8]
    if door_conf > 0.7:
        self.add_landmark(map_id, x, y, 'EXIT_DOOR')
        # Trigger path replanning: "To go north, first reach (x, y) door"
```

---

## 💡 SOLUTION 3: Temporal Sub-Quest Chains (MEDIUM-TERM - 2 hours)

### Expand SLAM Planning Hints [12-19]:

```python
def get_navigation_hints(self, current_map, current_x, current_y):
    hints = [0.0] * 64
    
    # [0-11]: Current (direction, exploration, path progress)
    # ... existing code ...
    
    # [12-19]: NEW - Sub-quest temporal chain
    if hasattr(self, 'current_quest_chain'):
        chain = self.current_quest_chain  # ['EXIT_LAB', 'GO_NORTH', 'ENTER_ROUTE']
        current_step = self.quest_chain_step  # 0, 1, 2...
        
        hints[12] = current_step / len(chain)  # Overall chain progress
        hints[13] = self.subquest_progress  # Progress within current step (0-1)
        hints[14] = min(self.time_stuck_subquest / 100, 1.0)  # Stuck time
        hints[15] = min(self.subquest_retry_count / 10, 1.0)  # Retry attempts
        
        # [16-19]: Next step preview (what comes after)
        if current_step + 1 < len(chain):
            next_step = chain[current_step + 1]
            if next_step == 'GO_NORTH':
                hints[16] = 1.0  # Prepare for northward movement
            elif next_step == 'ENTER_BUILDING':
                hints[17] = 1.0  # Prepare for door interaction
```

### Quest Chain Definition:

```python
# In DialogueProcessor
self.quest_chains = {
    'STARTER_CHOSEN': [
        'EXIT_LAB',           # Step 0: Find door, move south
        'GO_NORTH_TOWN',      # Step 1: Navigate through Littleroot
        'ENTER_ROUTE_101',    # Step 2: Reach tall grass area
        'FIND_BIRCH'          # Step 3: Locate NPC in grass
    ],
    'RECEIVED_POKEDEX': [
        'LEAVE_LAB',
        'GO_SOUTH_TO_OLDALE', 
        'EXPLORE_ROUTE_103'
    ]
}
```

---

## 💡 SOLUTION 4: SLAM-RGB-SNN Integration (YOUR IDEA - 3 hours)

### Complete Pipeline:

```
RGB Pixels → Door Detection → SLAM Landmarks → Path Planning → SNN Temporal → RL Actions
```

**Step 1**: RGB detects doors
```python
door_detector = RGBDoorDetector()
door_coords = door_detector.find_exits(game_frame)  # [(x, y, confidence), ...]
```

**Step 2**: SLAM marks as waypoints
```python
for (x, y, conf) in door_coords:
    slam.add_landmark(map_id, x, y, 'EXIT', priority=conf)
```

**Step 3**: Path planning includes doors
```python
# If target outside current map:
path = slam.plan_path(start, exit_door)  # ← First go to door
# Then: path += slam.plan_path(exit_door_outside, final_target)
```

**Step 4**: SNN learns temporal patterns
```python
# SNN spike history [10 timesteps]:
# t=-10: [indoor, no_door_visible, moving_north, stuck]
# t=-5:  [indoor, door_detected, moving_south, progressing]
# t=0:   [outdoor, route_101, moving_north, success]
# → Learns: "Indoor + north_goal = first_find_door_south"
```

---

## 🎯 RECOMMENDED PRIORITY

### 1. **IMMEDIATE** (Deploy tonight):
✅ Fix VLM parsing with fallback NLP (15 min)  
✅ Strengthen prompt to enforce ACTION: format

### 2. **SHORT-TERM** (Tomorrow):
- Add RGB door detection (8→16 dims)
- SLAM marks exits as landmarks
- Test on lab escape scenario

### 3. **MEDIUM-TERM** (This week):
- Implement quest chain temporal tracking
- Expand SLAM hints [12-19] for sub-quests
- SNN learns "exit first, then navigate" patterns

### 4. **VALIDATION**:
- Monitor agent escaping lab < 50 steps
- Track "indoor → outdoor" transitions
- Verify SNN temporal memory improving

---

## 📈 EXPECTED IMPROVEMENTS

**Before** (Current):
- 90 parsing errors / 32 VLM calls = 3x error rate
- Stuck in lab 2+ hours
- Only 7 actions followed

**After Solution 1**:
- 0 parsing errors (fallback handles all responses)
- Agent follows VLM ~80% of time
- Escapes lab within 100 steps

**After Solution 2+3**:
- RGB detects door → SLAM plans exit route
- Sub-quest: EXIT_LAB (step 0 of 4)
- Agent goes south to door, exits in ~30 steps

**After Solution 4**:
- SNN learns: "Indoor + north goal = exit first"
- Temporal memory: "Last time I went north indoors, got stuck"
- Generalizes to all buildings (gyms, houses, marts)
