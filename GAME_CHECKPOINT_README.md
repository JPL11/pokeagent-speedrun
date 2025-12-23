# Game State Checkpoint System

This system allows you to save and load the **emulator's game state** (not the RL model weights). This is useful for:
- Creating restore points at important game moments
- Testing different strategies from the same starting point
- Recovering from bad situations without restarting from scratch

## Quick Usage

### Save Current Game State

```bash
# Save with automatic timestamped filename
python save_game_checkpoint.py

# Save to specific file
python save_game_checkpoint.py my_checkpoint.state

# Save to specific location
python save_game_checkpoint.py checkpoints/before_rival_battle.state
```

### Load a Saved Game State

```bash
# Load from specific file
python save_game_checkpoint.py --load .pokeagent_cache/game_checkpoint_20251217_072743.state

# Or use -l shorthand
python save_game_checkpoint.py -l checkpoints/before_rival_battle.state
```

## What Gets Saved?

When you save a game state, it includes:
- **Emulator RAM** - Complete game memory (player position, inventory, flags, etc.)
- **Milestones** - Completed milestones (saved to `.milestones` file)
- **Map data** - Persistent location grids (saved to `.grids` file)

This is a **complete snapshot** of the game at that exact moment.

## Storage Location

By default, checkpoints are saved to `.pokeagent_cache/` directory with timestamp:
```
.pokeagent_cache/
  game_checkpoint_20251217_072743.state       # Main state file
  game_checkpoint_20251217_072743.state.milestones  # Milestone data
  game_checkpoint_20251217_072743.state.grids        # Map data
```

## Important Notes

⚠️ **This saves the GAME state, not the RL model weights!**
- The RL model continues training from its current policy
- Loading a game state doesn't reset what the agent has learned
- It only resets the game to that checkpoint position

⚠️ **Server must be running**
- The game server (port 8000) must be active
- Run `python server/app.py` first if it's not running

## Use Cases

### 1. Before Important Battles
```bash
# Right before challenging rival on Route 103
python save_game_checkpoint.py checkpoints/pre_rival_battle.state
```

### 2. After Key Milestones
```bash
# After getting starter Pokemon
python save_game_checkpoint.py checkpoints/got_starter.state

# After beating rival
python save_game_checkpoint.py checkpoints/post_rival.state
```

### 3. Testing Different Paths
```bash
# Save at decision point
python save_game_checkpoint.py test_point.state

# Try path A...
# If it doesn't work, reload:
python save_game_checkpoint.py --load test_point.state

# Try path B...
```

## API Endpoints

The underlying server API endpoints:

### Save State
```bash
curl -X POST http://localhost:8000/save_state \
  -H "Content-Type: application/json" \
  -d '{"filepath": "my_save.state"}'
```

### Load State
```bash
curl -X POST http://localhost:8000/load_state \
  -H "Content-Type: application/json" \
  -d '{"filepath": "my_save.state"}'
```

## Troubleshooting

**"Could not connect to game server"**
- Make sure the server is running: `python server/app.py`

**"File not found"**
- Check the filepath is correct
- Use absolute path or relative to current directory

**State loads but game looks wrong**
- The state file might be corrupted
- Try loading a different checkpoint
- Restart the server and try again
