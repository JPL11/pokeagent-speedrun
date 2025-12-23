# Identity Learning System - Architecture Proposal

**Date**: November 23, 2025  
**Problem**: Persistent hallucination where VLM confuses player identity (thinks player is Birch, May instead of Brendan, wrong house, etc.)  
**Root Cause**: Hardcoded assumptions instead of reading actual game state

---

## Current Issues

### 1. **Avatar Confusion**
- VLM sometimes thinks player is Professor Birch
- Confuses player sprite with NPC sprites
- Cannot distinguish player-controlled character from other characters on screen

### 2. **Gender/Name Confusion**  
- Player chooses gender at start (male=Brendan, female=May)
- Rival is opposite gender (male player → May rival, female player → Brendan rival)
- Current system uses milestone location inference (fragile, breaks on reset)
- Fails when milestone data is stale or missing

### 3. **Dialog State Confusion**
- VLM sometimes thinks player IS the one speaking instead of listening
- Confuses "You" in dialog text with player identity vs NPC reference

---

## Solution: Memory-Based Identity Learning

### Architecture: Identity Context Provider

Instead of hardcoding rules, **read identity directly from game memory** and provide it as structured context to VLM.

```
Game Memory → Identity Context Provider → VLM Prompt
     ↓               ↓                         ↓
0x0202573B      player_gender=0           "You are Brendan (male)"
0x02025734      player_name="RED"         "Your name is RED"
gSprites[0]     player_sprite_id=1        "Your avatar: male sprite"
```

---

## Implementation Plan

### Phase 1: Memory Reading (IMMEDIATE)

**File**: `pokemon_env/memory_reader.py`

Add identity reading methods:

```python
class MemoryAddresses:
    PLAYER_GENDER = 0x0202573B  # 0=male(Brendan), 1=female(May)
    PLAYER_NAME = 0x02025734     # 7-character name string
    # Add sprite detection addresses

class MemoryReader:
    def get_player_identity(self):
        """Read player identity from memory - GROUND TRUTH
        
        Returns:
            dict: {
                'gender': 0 or 1,           # 0=male, 1=female
                'gender_name': 'male' or 'female',
                'character': 'Brendan' or 'May',
                'rival_character': 'May' or 'Brendan',
                'name': str,                # Player's chosen name
                'sprite_id': int            # Player sprite graphics ID
            }
        """
        try:
            gender = self._read_u8(self.addresses.PLAYER_GENDER)
            name_bytes = self._read_bytes(self.addresses.PLAYER_NAME, 7)
            name = self._decode_pokemon_text(name_bytes)
            
            character = 'Brendan' if gender == 0 else 'May'
            rival = 'May' if gender == 0 else 'Brendan'
            
            return {
                'gender': gender,
                'gender_name': 'male' if gender == 0 else 'female',
                'character': character,
                'rival_character': rival,
                'name': name,
                'verification': 'Read from memory 0x0202573B'
            }
        except Exception as e:
            logger.error(f"Failed to read player identity: {e}")
            return None
```

### Phase 2: Server API Exposure

**File**: `server/app.py`

Add `/identity` endpoint:

```python
@app.get("/identity")
def get_player_identity():
    """Return player identity from memory"""
    identity = memory_reader.get_player_identity()
    return identity or {"error": "Could not read identity"}
```

Update `/state` endpoint to include identity:

```python
@app.get("/state")
def get_state():
    state = memory_reader.get_state()
    identity = memory_reader.get_player_identity()
    return {
        **state,
        'player': {
            'name': identity['name'],
            'character': identity['character'],
            'gender': identity['gender_name']
        } if identity else None
    }
```

### Phase 3: VLM Integration (CRITICAL)

**File**: `train_rl_vlm_screen_aware.py`

Replace hardcoded identity assumptions with memory reads:

```python
class ServerAPIOracle:
    def get_player_identity(self):
        """Query server for player identity - GROUND TRUTH"""
        try:
            response = requests.get(f"{self.server_url}/identity", timeout=5)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.warning(f"Could not get player identity: {e}")
        return None
    
    def get_query(self, frame, memory):
        # Get ACTUAL identity from game memory
        identity = self.get_player_identity()
        
        if not identity:
            logger.error("⚠️ No identity data - VLM may hallucinate!")
            identity_context = "⚠️ Player identity unknown"
        else:
            identity_context = f"""
🎭 YOUR IDENTITY (from game memory):
• You are: {identity['character']} ({identity['gender_name']})
• Your name: {identity['name']}
• Your rival: {identity['rival_character']}
• Verification: {identity['verification']}

⚠️ CRITICAL: You are {identity['character']}, NOT {identity['rival_character']}!
⚠️ You are NOT Professor Birch! Birch is an NPC who gives you Pokemon.
"""
        
        prompt = f"""Pokemon Emerald Speedrun - Story Progress

{identity_context}

📚 **Professor Birch** is a Pokemon researcher (NPC) who gives you your first Pokemon
👥 **Your Rival** ({identity['rival_character'] if identity else 'May/Brendan'}) is Birch's child

...rest of prompt...
"""
```

### Phase 4: House Identity Fix

Replace milestone inference with memory-based logic:

```python
def _get_house_identity(self, location_name, completed):
    """Use player identity from memory, not milestone inference"""
    
    # Get GROUND TRUTH from memory
    identity = self.oracle.get_player_identity()
    
    if not identity:
        return None  # Can't determine without memory
    
    player_character = identity['character']  # 'Brendan' or 'May'
    rival_character = identity['rival_character']
    
    # Simple logic based on ACTUAL gender
    if "BRENDANS_HOUSE" in location_name:
        if player_character == 'Brendan':
            return {
                'is_players_house': True,
                'player_character': 'Brendan',
                'rival_character': 'May',
                'target_house': "MAY'S HOUSE",
                'verification': f'Player gender from memory: male'
            }
        else:
            return {
                'is_players_house': False,
                'is_rivals_house': True,
                'player_character': 'May',
                'rival_character': 'Brendan',
                'target_house': None,
                'verification': f'Player gender from memory: female'
            }
    
    elif "MAYS_HOUSE" in location_name:
        if player_character == 'May':
            return {
                'is_players_house': True,
                'player_character': 'May',
                'rival_character': 'Brendan',
                'target_house': "BRENDAN'S HOUSE",
                'verification': f'Player gender from memory: female'
            }
        else:
            return {
                'is_players_house': False,
                'is_rivals_house': True,
                'player_character': 'Brendan',
                'rival_character': 'May',
                'target_house': None,
                'verification': f'Player gender from memory: male'
            }
    
    return None
```

---

## Phase 5: Sprite-Based Identity (ADVANCED)

For visual verification, detect which sprite is player-controlled:

```python
class MemoryReader:
    def get_player_sprite_id(self):
        """Get player's sprite graphics ID for visual identification
        
        The player sprite has a specific graphics ID that differs from NPCs.
        Male player (Brendan): graphics_id = 0x01
        Female player (May): graphics_id = 0x02
        """
        try:
            # Player object is typically first in object events
            player_obj_addr = 0x02036E00  # gObjectEvents[0]
            graphics_id = self._read_u8(player_obj_addr + 0x00)
            
            return {
                'sprite_id': graphics_id,
                'character': 'Brendan' if graphics_id == 0x01 else 'May',
                'verification': 'Player sprite graphics_id'
            }
        except Exception as e:
            logger.debug(f"Could not read sprite ID: {e}")
            return None
```

Then VLM can verify:
```
"Your avatar on screen is the MALE sprite (Brendan), not the female sprite (May)."
```

---

## Benefits of Memory-Based Approach

| Aspect | Milestone Inference ❌ | Memory Reading ✅ |
|--------|------------------------|-------------------|
| **Accuracy** | Fails if milestone data stale | Always correct (ground truth) |
| **Robustness** | Breaks on episode reset | Works from frame 1 |
| **Simplicity** | Complex fallback logic | Single memory read |
| **Verifiable** | Inferred from behavior | Direct from game state |
| **Real-time** | Requires milestone completion | Available immediately |
| **Debugging** | Hard to trace inference | Easy to verify: log memory value |

---

## Testing Plan

1. **Unit Tests**: Test `get_player_identity()` with male/female saves
2. **Integration Test**: Start new game, choose male, verify identity before any milestones
3. **Hallucination Test**: Check VLM prompt includes correct identity at step 1
4. **House Test**: Enter Brendan's house, verify system correctly identifies as player's house
5. **Reset Test**: Reset episode, verify identity persists (not dependent on milestone_tracking)

---

## Migration Path

### Immediate (Tonight):
1. ✅ Add `get_player_identity()` to memory_reader.py
2. ✅ Add `/identity` endpoint to server
3. ✅ Update VLM prompt to include identity context at TOP

### Tomorrow:
4. Replace milestone inference in `_get_house_identity()`
5. Add sprite-based visual verification
6. Test with fresh game (male and female)

### Future:
7. Use identity for ALL story context (not just houses)
8. Add identity to observation space for CNN/SNN learning
9. Log identity at every VLM call for debugging

---

## Expected Impact

**Before**: "🚨 HOUSE IDENTITY VERIFICATION: Awaiting PLAYER_BEDROOM location data... You are May"  
**After**: "🎭 YOUR IDENTITY: You are Brendan (male) from memory 0x0202573B. Rival is May."

**Before**: VLM says "rival May" when in Brendan's house (wrong if player is male)  
**After**: VLM reads "You are Brendan" → knows Brendan's house = player's house

**Before**: Hallucination persists because inference is wrong  
**After**: Memory reading provides ground truth, hallucination impossible

---

## Implementation Priority

**CRITICAL (Tonight):**
- `get_player_identity()` in memory_reader.py
- `/identity` endpoint in server
- Identity context in VLM prompt (top position)

**HIGH (Tomorrow):**
- Replace House Identity Verifier milestone inference
- Test with male/female characters
- Verify no more hallucination

**MEDIUM (Next week):**
- Sprite-based visual verification
- Add to observation space for learning

---

This is the **architectural** solution - learn from memory, don't hardcode assumptions.
