# Enhanced Terrain Reasoning for Route 101 Navigation

## Problem
Agent stuck on Route 101 because it doesn't understand:
1. **One-way ledges** - Can only jump DOWN, not climb up
2. **Tall grass is walkable** - Agent avoids grass thinking it's blocked
3. **Directional terrain** - Some tiles only passable in specific directions

## Solution: Enhanced SNN Input with Terrain Features

### Current SNN Inputs (320 neurons):
```python
snn_input = [
    vlm_goals (64),      # High-level objectives
    milestones (64),     # Story flags
    quest_memory (64),   # Dialog-based quests
    cnn_vision (64),     # Local pixels
    grid_map (64)        # Global position
]
```

### NEW: Add Terrain Reasoning Input (6th stream, 64 neurons):
```python
terrain_features = {
    # Immediate surroundings (4 directions × 4 features = 16 dims)
    'north': [is_blocked, is_grass, is_ledge, ledge_direction],
    'south': [is_blocked, is_grass, is_ledge, ledge_direction],
    'east':  [is_blocked, is_grass, is_ledge, ledge_direction],
    'west':  [is_blocked, is_grass, is_ledge, ledge_direction],
    
    # Current tile features (8 dims)
    'on_grass': 1.0,           # Standing on tall grass
    'on_ledge': 0.0,           # Standing on ledge
    'on_water': 0.0,           # Need surf
    'on_door': 0.0,            # At entrance
    'can_jump_north': 0.0,     # Ledge allows north jump
    'can_jump_south': 1.0,     # Ledge allows south jump
    'can_jump_east': 0.0,
    'can_jump_west': 0.0,
    
    # Movement history (8 dims)
    'stuck_on_ledge': 0.8,     # Tried to go wrong way on ledge
    'avoided_grass': 0.3,      # Agent avoiding grass unnecessarily
    'repeated_blocked': 0.5,   # Hitting walls repeatedly
    'backtracking': 0.2,       # Going back and forth
    
    # Pathfinding hints (32 dims)
    'goal_direction': [0.0, 1.0, 0.0, 0.0],  # [N, S, E, W] to goal
    'walkable_escape_routes': 3,              # Number of valid exits
    'nearest_progress_tile': 12.0,            # Distance to advancement
    # ... (padding to 64 total)
}
```

## Key Terrain Behaviors (from Pokemon Emerald)

### Ledges (One-Way)
```python
JUMP_NORTH = 58    # Can ONLY move TO this from SOUTH (jump down northward)
JUMP_SOUTH = 59    # Can ONLY move TO this from NORTH (jump down southward)  
JUMP_EAST = 56     # Can ONLY move TO this from WEST
JUMP_WEST = 57     # Can ONLY move TO this from EAST
```

**Critical**: You CAN'T climb UP ledges! Agent must find alternate routes.

### Grass (Walkable but triggers encounters)
```python
TALL_GRASS = 2     # Walkable! Just triggers wild Pokemon
LONG_GRASS = 3     # Also walkable
SHORT_GRASS = 7    # Walkable, no encounters
```

**Agent should NOT avoid grass** - it's the main path through Route 101!

### Water (Need Surf HM)
```python
DEEP_WATER = 18    # BLOCKED until you get Surf HM
OCEAN_WATER = 21   # BLOCKED
SHALLOW_WATER = 23 # BLOCKED
```

## Implementation Plan

### Step 1: Extract Terrain Features from Game State
```python
class TerrainFeatureExtractor:
    def __init__(self):
        self.ledge_stuck_count = 0
        self.grass_avoidance_count = 0
        self.last_positions = []
    
    def extract_features(self, state_data):
        map_tiles = state_data.get('map', {}).get('tiles', [])
        player_pos = state_data.get('player', {}).get('position', {})
        
        # Get 4 adjacent tiles
        adjacent = self._get_adjacent_tiles(map_tiles, player_pos)
        
        # Current tile behavior
        current_tile = self._get_current_tile(map_tiles, player_pos)
        
        # Build 64-dim terrain encoding
        terrain_vec = []
        
        # Adjacent tiles (4 dirs × 4 features = 16)
        for direction in ['north', 'south', 'east', 'west']:
            tile = adjacent[direction]
            terrain_vec.extend([
                1.0 if self._is_blocked(tile) else 0.0,
                1.0 if self._is_grass(tile) else 0.0,
                1.0 if self._is_ledge(tile) else 0.0,
                self._get_ledge_direction_encoding(tile, direction)
            ])
        
        # Current tile state (8)
        terrain_vec.extend([
            1.0 if self._is_grass(current_tile) else 0.0,
            1.0 if self._is_ledge(current_tile) else 0.0,
            1.0 if self._is_water(current_tile) else 0.0,
            1.0 if self._is_door(current_tile) else 0.0,
            1.0 if self._can_jump_north(current_tile) else 0.0,
            1.0 if self._can_jump_south(current_tile) else 0.0,
            1.0 if self._can_jump_east(current_tile) else 0.0,
            1.0 if self._can_jump_west(current_tile) else 0.0,
        ])
        
        # Movement patterns (8)
        terrain_vec.extend([
            self._compute_ledge_stuck_signal(),
            self._compute_grass_avoidance_signal(),
            self._compute_wall_collision_signal(),
            self._compute_backtracking_signal(),
            0.0, 0.0, 0.0, 0.0  # reserved
        ])
        
        # Pathfinding hints (32) - pad to 64 total
        goal_dir = self._compute_goal_direction(state_data)
        terrain_vec.extend(goal_dir)  # 4 dims
        terrain_vec.extend([0.0] * 28)  # padding
        
        return terrain_vec[:64]  # Ensure exactly 64 dims
```

### Step 2: Add to SNN Architecture
```python
# In StoryDrivenCNNSNNExtractor.__init__()
self.terrain_feature_encoder = nn.Sequential(
    nn.Linear(64, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.Tanh()
)

# SNN now has 6 inputs × 64 = 384 neurons
snn_hidden_dim = 384
```

### Step 3: Enhanced Reward Shaping
```python
# Reward walking through grass (not avoiding it!)
if current_tile_is_grass and agent_moved_into_grass:
    reward += 1.0  # Good! Progress through Route 101
    
# Penalize trying to climb ledges
if trying_to_move_against_ledge_direction:
    reward -= 0.5  # Learn that ledges are one-way
    
# Reward finding alternate routes around ledges
if found_walkable_path_around_ledge:
    reward += 2.0  # Smart navigation!
```

## Route 101 Specific Logic

Route 101 layout has:
- **Tall grass patches** in the middle (WALKABLE)
- **Ledges on edges** (ONE-WAY DOWN)
- **Trees on borders** (IMPASSABLE)

Agent needs to learn:
1. Walk THROUGH grass (don't avoid it)
2. Can't climb UP ledges (find alternate route)
3. Trees are truly blocked (don't waste time)

## Expected Behavior After Training

### Before (Stuck):
```
Agent: *sees grass* "That looks blocked!"
Agent: *tries ledge* "Why can't I go up?"
Agent: *loops in same spot*
```

### After (Smart):
```
Agent: "Grass is walkable, just triggers encounters"
Agent: "This ledge only goes DOWN, I need to find another path"
Agent: "I'll walk through the grass to reach the north exit"
```

## Testing Plan

1. **Start at Route 101 entrance** (from Littleroot)
2. **Goal**: Reach Oldale Town (north exit)
3. **Metrics**:
   - Time to reach Oldale: < 200 steps (vs current: stuck forever)
   - Grass entries: > 5 (proves agent walks through grass)
   - Ledge collision attempts: < 3 (proves agent learned one-way)
   - Unique tiles explored: > 30 (proves full map coverage)

## Integration Checklist

- [ ] Implement `TerrainFeatureExtractor` class
- [ ] Add terrain encoder to SNN (6th input stream)
- [ ] Update SNN dimensions: 320 → 384 neurons
- [ ] Add terrain-aware reward shaping
- [ ] Test on Route 101 specifically
- [ ] Verify agent walks through grass
- [ ] Verify agent respects ledge directions
- [ ] Monitor training logs for terrain understanding
