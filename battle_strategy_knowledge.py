# Pokemon Emerald Battle Strategy Knowledge Base
# Based on competitive guides and speedrun strategies
# Fed to SNN/CNN/VLM for temporal battle reasoning

BATTLE_STRATEGY_GUIDE = {
    "type_effectiveness": {
        # Super effective (2x damage)
        "WATER": ["FIRE", "GROUND", "ROCK"],
        "FIRE": ["GRASS", "ICE", "BUG", "STEEL"],
        "GRASS": ["WATER", "GROUND", "ROCK"],
        "ELECTRIC": ["WATER", "FLYING"],
        "FIGHTING": ["NORMAL", "ICE", "ROCK", "DARK", "STEEL"],
        "PSYCHIC": ["FIGHTING", "POISON"],
        "DARK": ["PSYCHIC", "GHOST"],
        "GHOST": ["PSYCHIC", "GHOST"],
        "DRAGON": ["DRAGON"],
        "ICE": ["GRASS", "GROUND", "FLYING", "DRAGON"],
        "ROCK": ["FIRE", "ICE", "FLYING", "BUG"],
        "GROUND": ["FIRE", "ELECTRIC", "POISON", "ROCK", "STEEL"],
        "FLYING": ["GRASS", "FIGHTING", "BUG"],
        "POISON": ["GRASS"],
        "BUG": ["GRASS", "PSYCHIC", "DARK"],
        "STEEL": ["ICE", "ROCK"],
        
        # Not very effective (0.5x damage)
        "WATER_WEAK": ["WATER", "GRASS", "DRAGON"],
        "FIRE_WEAK": ["FIRE", "WATER", "ROCK", "DRAGON"],
        "GRASS_WEAK": ["FIRE", "GRASS", "POISON", "FLYING", "BUG", "DRAGON", "STEEL"],
        "ELECTRIC_WEAK": ["ELECTRIC", "GRASS", "DRAGON"],
        # ... (add more as needed)
    },
    
    "early_game_strategy": {
        # Route 101-103 (Levels 2-5)
        "route_101": {
            "wild_encounters": ["POOCHYENA", "ZIGZAGOON", "WURMPLE"],
            "levels": [2, 3, 4],
            "strategy": [
                "Avoid wild battles unless grinding",
                "Poochyena uses Tackle (Normal), Zigzagoon uses Tackle",
                "Your starter (Lv5) should one-shot everything",
                "Don't waste PP on weak Pokemon"
            ],
            "escape_priority": "HIGH - Don't waste time on weak Pokemon"
        },
        
        "rival_battle_route_103": {
            "opponent": "May/Brendan",
            "pokemon": {
                "if_you_chose_treecko": {"species": "TORCHIC", "level": 5, "type": "FIRE"},
                "if_you_chose_torchic": {"species": "MUDKIP", "level": 5, "type": "WATER"},
                "if_you_chose_mudkip": {"species": "TREECKO", "level": 5, "type": "GRASS"}
            },
            "strategy": [
                "This is a TYPE MATCHUP test!",
                "If disadvantage: Use stat moves (Growl/Tail Whip) then attack",
                "If advantage: Spam your STAB move (Scratch, Ember, Pound)",
                "DO NOT RUN - This unlocks Pokedex milestone"
            ],
            "win_condition": "Defeat rival to get Pokedex from Birch",
            "loss_recovery": "Blackout teleports to Pokemon Center (Oldale Town), re-navigate to Route 103"
        },
        
        "rustboro_gym_roxanne": {
            "gym_leader": "Roxanne",
            "team": [
                {"species": "GEODUDE", "level": 12, "type": ["ROCK", "GROUND"], 
                 "moves": ["TACKLE", "DEFENSE_CURL", "ROCK_THROW"]},
                {"species": "GEODUDE", "level": 12, "type": ["ROCK", "GROUND"],
                 "moves": ["TACKLE", "DEFENSE_CURL", "ROCK_THROW"]},
                {"species": "NOSEPASS", "level": 15, "type": ["ROCK"],
                 "moves": ["TACKLE", "HARDEN", "ROCK_THROW", "ROCK_TOMB"]}
            ],
            "strategy": [
                "CRITICAL: Need Water or Grass type (4x effective on Geodude)",
                "If you have Mudkip: WATER GUN destroys everything",
                "If you have Treecko: ABSORB is super effective",
                "If you have Torchic: GRIND to Lv16 for DOUBLE_KICK (Fighting type)",
                "Rock Tomb LOWERS SPEED - switch if hit",
                "Recommended level: 13-15"
            ],
            "preparation": [
                "Catch Lotad (Water/Grass) in Route 102 if you have Torchic",
                "Stock up on Potions (3-5)",
                "Consider catching Makuhita (Fighting type) in Granite Cave"
            ],
            "loss_recovery": "Blackout → Pokemon Center. LEARN: Need type advantage. Go catch counter-type Pokemon before retry."
        }
    },
    
    "battle_tactics": {
        "general_rules": [
            "Type advantage = 2x damage (super effective)",
            "Type disadvantage = 0.5x damage (not very effective)", 
            "STAB (Same Type Attack Bonus) = 1.5x damage",
            "Critical hits = 2x damage (random, ~6% chance)",
            "Status moves (Growl, Leer) reduce stats - use when stalling"
        ],
        
        "hp_management": {
            "critical_threshold": 0.25,  # Below 25% HP
            "safe_threshold": 0.6,  # Above 60% HP
            "actions": {
                "hp_below_25": "Consider switching or using Potion",
                "hp_below_50": "Play defensively, prioritize super-effective moves",
                "hp_above_60": "Aggressive strategy, set up stat moves if needed"
            }
        },
        
        "when_to_escape": [
            "Wild Pokemon 5+ levels higher than yours",
            "Your Pokemon below 30% HP and no potions",
            "Opponent has type advantage and you have no counter",
            "Already caught this species (no need to battle)",
            "Low on PP for important moves"
        ],
        
        "when_to_fight": [
            "Trainer battle (can't escape, mandatory)",
            "Rare Pokemon encounter (Ralts, Abra, etc.)",
            "Your Pokemon needs EXP (within 3 levels)",
            "Opponent has type disadvantage (easy win)",
            "Gym Leader / Story boss (required for progression)"
        ]
    },
    
    "blackout_recovery_protocol": {
        "what_happens": [
            "All Pokemon fainted → Blackout",
            "Teleport to last visited Pokemon Center",
            "Lose 50% of money",
            "All Pokemon revived to 1 HP",
            "Reset position (lose progress)"
        ],
        
        "after_blackout_actions": [
            "1. ANALYZE WHY: Under-leveled? Bad type matchup? No items?",
            "2. HEAL TEAM: Talk to Nurse Joy (automatic full heal)",
            "3. PREPARE: Buy Potions, catch counter-type Pokemon",
            "4. GRIND: Battle wild Pokemon to gain 2-3 levels",
            "5. RE-ATTEMPT: Return to battle location with better strategy"
        ],
        
        "prevention": [
            "Always keep party HP above 50% before gym/rival battles",
            "Stock 3-5 Potions before major battles",
            "Check type matchups - catch counter Pokemon if needed",
            "Don't fight trainers with low HP (heal first)",
            "Save PP for boss battles (don't spam on wild Pokemon)"
        ]
    },
    
    "experience_learning": {
        # What SNN should learn from repeated failures
        "temporal_patterns": {
            "pattern_1": {
                "sequence": ["ENTER_GYM", "BATTLE_LEADER", "FAINT", "BLACKOUT", "POKEMON_CENTER"],
                "lesson": "Under-leveled or bad type matchup",
                "solution": "Before re-entering gym: Grind 2-3 levels OR catch counter-type"
            },
            "pattern_2": {
                "sequence": ["LOW_HP", "FIGHT_TRAINER", "FAINT"],
                "lesson": "Fought with critical HP",
                "solution": "If HP < 50%, avoid trainers OR use Potion BEFORE battle"
            },
            "pattern_3": {
                "sequence": ["WATER_MOVE", "vs_FIRE", "WIN"] * 3,
                "lesson": "Type advantage learned through experience",
                "solution": "Reinforce: Water beats Fire in memory"
            },
            "pattern_4": {
                "sequence": ["ROXANNE", "FAINT"] * 2,
                "lesson": "Specific gym leader requires specific strategy",
                "solution": "Meta-knowledge: Need Water/Grass for Roxanne. Go catch Lotad."
            }
        }
    },
    
    "speedrun_knowledge": {
        # Advanced strategies from speedrunners
        "optimal_party": {
            "solo_run": "One overleveled Pokemon (Mudkip → Swampert recommended)",
            "reason": "Fewer Pokemon = More EXP per battle = Faster leveling",
            "drawback": "Blackout = lose more progress"
        },
        
        "skip_wild_battles": "Run from all wild Pokemon unless grinding is needed",
        "trainer_battles_mandatory": "Cannot escape, must win or blackout",
        
        "gym_order": [
            "Rustboro (Roxanne - Rock)",
            "Dewford (Brawly - Fighting)", 
            "Mauville (Wattson - Electric)",
            "Lavaridge (Flannery - Fire)",
            "Petalburg (Norman - Normal)",
            "Fortree (Winona - Flying)",
            "Mossdeep (Tate & Liza - Psychic)",
            "Sootopolis (Juan - Water)"
        ],
        
        "level_checkpoints": {
            "rival_route_103": 5,
            "roxanne": 15,
            "brawly": 20,
            "wattson": 25,
            "flannery": 30,
            "norman": 32,
            "winona": 35,
            "tate_liza": 45,
            "juan": 47,
            "elite_four": 55
        }
    }
}


def encode_battle_knowledge_for_snn(battle_state, agent_history, blackout_count):
    """
    Convert battle strategy knowledge into 64-dim vector for SNN temporal reasoning.
    This allows agent to learn from past failures and apply meta-knowledge.
    
    Returns: [64-dim] encoding of battle context + learned patterns
    """
    encoding = [0.0] * 64
    
    # [0-7]: Type effectiveness memory (did I learn type matchups?)
    # Example: Water beat Fire 3 times → encoding[0] = 0.3 (confidence)
    type_effectiveness_memory = agent_history.get('type_effectiveness_wins', {})
    for i, type_pair in enumerate([('WATER', 'FIRE'), ('FIRE', 'GRASS'), ('GRASS', 'WATER')]):
        if i < 8:
            wins = type_effectiveness_memory.get(f"{type_pair[0]}_vs_{type_pair[1]}", 0)
            encoding[i] = min(wins / 10.0, 1.0)  # Confidence grows with experience
    
    # [8-15]: Blackout history (temporal penalty signal)
    encoding[8] = min(blackout_count / 10.0, 1.0)  # Total blackouts
    encoding[9] = 1.0 if blackout_count > 0 else 0.0  # Ever blacked out?
    encoding[10] = agent_history.get('blackouts_last_100_steps', 0) / 5.0  # Recent failures
    
    # [16-23]: Current battle risk assessment
    if battle_state:
        player_hp_fraction = battle_state.get('player_hp', 1.0)
        opponent_level = battle_state.get('opponent_level', 0)
        player_level = battle_state.get('player_level', 0)
        
        encoding[16] = player_hp_fraction  # 0.0 = critical, 1.0 = full
        encoding[17] = 1.0 if player_hp_fraction < 0.25 else 0.0  # Critical HP flag
        encoding[18] = max(0, opponent_level - player_level) / 10.0  # Level disadvantage
        encoding[19] = 1.0 if opponent_level > player_level + 3 else 0.0  # High risk
    
    # [24-31]: Learned gym strategies (did I beat this gym before?)
    gym_victories = agent_history.get('gym_victories', set())
    encoding[24] = 1.0 if 'ROXANNE' in gym_victories else 0.0
    encoding[25] = 1.0 if 'BRAWLY' in gym_victories else 0.0
    # ... (encode more gyms)
    
    # [32-39]: Item availability (do I have healing items?)
    encoding[32] = min(agent_history.get('potion_count', 0) / 5.0, 1.0)
    encoding[33] = 1.0 if agent_history.get('has_revive', False) else 0.0
    
    # [40-47]: Temporal failure patterns (SNN learns from repetition)
    # Did I die to same trainer 2+ times? → Need different strategy
    recent_deaths = agent_history.get('recent_death_locations', [])
    if len(recent_deaths) >= 2 and recent_deaths[-1] == recent_deaths[-2]:
        encoding[40] = 1.0  # Repeated failure flag
        encoding[41] = len([x for x in recent_deaths if x == recent_deaths[-1]]) / 5.0
    
    # [48-55]: Meta-knowledge activation (guide-based strategies)
    current_location = battle_state.get('location', '')
    if 'RUSTBORO_GYM' in current_location:
        # Activate Roxanne strategy: Need Water/Grass
        encoding[48] = 1.0  # Roxanne encounter
        has_counter = battle_state.get('party_has_water_or_grass', False)
        encoding[49] = 1.0 if has_counter else 0.0
    
    # [56-63]: Reserved for online guide integration
    # Future: Scrape Bulbapedia/Serebii for real-time strategy hints
    
    return encoding
