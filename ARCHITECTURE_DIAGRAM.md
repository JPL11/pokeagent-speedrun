# PokéAgent System Architecture (Detailed Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        POKÉAGENT SYSTEM ARCHITECTURE                        │
│                      Three-Tier Strategic RL Framework                      │ 
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: VLM STRATEGIC LAYER (High-Level Reasoning)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────────┐               │
│  │  Game Frame  │───▶│   ServerAPIOracle (VLM Interface)    │                │
│  │  (240x160)   │    │  • Multi-modal prompt construction   │               │
│  └──────────────┘    │  • 4-case battle guidance system     │               │
│                      │  • Stuck detection & escape planning │               │
│  ┌──────────────┐    │  • Action parsing with validation    │               │
│  │ Server State │───▶│                                      │               │
│  │ /state API   │    │  Backends: OpenAI, Anthropic,        │               │
│  └──────────────┘    │            Google, OpenRouter        │               │
│                      └────────────┬─────────────────────────┘               │
│  ┌──────────────┐                 │                                         │
│  │ Episodic     │─────────────────┘                                         │
│  │ Memory       │    Output: Strategic Directive + Guidance Text            │
│  └──────────────┘                 │                                         │
│                                   ▼                                         │
│              ┌─────────────────────────────────────────────┐                │
│              │   Strategic Directive Manager               │                │
│              │   • Evaluates party level vs milestone      │                │
│              │   • Sets directive: GRIND / PROCEED / HEAL  │                │
│              │   • Tracks grinding progress & transitions  │                │
│              │   • Feedback loop: target reached → update  │                │
│              └─────────────────────────────────────────────┘                │
│                                    │                                        │
│              Directives: GRIND, PROCEED, HEAL, POST_BLACKOUT,               │
│                         PREPARE_GYM, GYM_BATTLE, OPTIONAL_GRIND             │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │ Strategic Context
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TIER 2: SNN TEMPORAL LAYER (Pattern Learning)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │                      SNNEncoder (Spiking Neural Network)           │     │
│  │                                                                    │     │
│  │   9 Input Streams (all active in current build):                   │     │
│  │   1. VLM High-Level Goals        (strategic directives, 64 neurons)│     │
│  │   2. Milestone / Story Flags     (quest progress, 64 neurons)      │     │
│  │   3. Structured Quest Memory     (parsed NPC dialogue, 64 neurons) │     │
│  │   4. CNN Local Vision            (immediate surroundings, 64 neurons)    │
│  │   5. Grid / Map Global Position  (town/route ID, 64 neurons)       │     │
│  │   6. Terrain Feature Encoder     (server metatiles, 64 neurons)    │     │
│  │   7. RGB Terrain Analyzer        (pixel heuristics, 64 neurons)    │     │
│  │   8. SLAM Path Planning          (A* hints, 64 neurons)            │     │
│  │   9. Episodic Memory k-NN        (action hist + confidence, 32)    │     │
│  │      • Confidence ≥0.7 triggers memory spike + SLAM/VLM attenuation│     │
│  │                                                                    │     │
│  │  Total Input: 544 neurons (8×64 + 1×32)                            │     │
│  │                                                                    │     │
│  │  ┌─────────────┐   ┌─────────────┐   ┌──────────────┐              │     │
│  │  │ LIF Neurons │──▶│ Spike Timing│──▶│ Temporal     │             │     │
│  │  │ (544 units) │   │  History    │   │  Features    │              │     │
│  │  └─────────────┘   └─────────────┘   └──────────────┘              │     │
│  │                                                                    │     │
│  │  Output: 544-dim temporal features (mean of 10-timestep spike history)│     │
│  │   • Full spike history aggregation (not 5 hand-crafted features)   │     │
│  │   • Temporal memory: "what happened in last 10 timesteps?"         │     │
│  │   • Learnable LIF dynamics: threshold & decay parameters           │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                    │                                        │
│                                    │ SNN Temporal Features (544-dim)         │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FEATURE EXTRACTION LAYER (Multi-Modal Fusion)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐      │
│  │            StoryDrivenCNNSNNExtractor (512-dim output)            │      │
│  │                                                                   │      │
│  │  ┌──────────────────────────────────────────────────────────────┐  │      │
│  │  │                    CNN Branch (Spatial)                      │  │      │
│  │  │  • Conv2d 32 (8×8 kernel, stride 4)                         │  │      │
│  │  │  • Conv2d 64 (4×4 kernel, stride 2)                         │  │      │
│  │  │  • Conv2d 128 (3×3 kernel)                                   │  │      │
│  │  │  • Conv2d 128 (3×3 kernel)                                   │  │      │
│  │  │  • Flatten → ~12,800 features (for 4-frame stack)            │  │      │
│  │  └──────────────────────┬───────────────────────────────────────┘  │      │
│  │                         │                                           │      │
│  │                         ▼                                           │      │
│  │  ┌──────────────────────────────────────────────────────────────┐  │      │
│  │  │         FEEDBACK LOOP: SNN → CNN Attention Modulation         │  │      │
│  │  │  • SNN temporal features (544-dim) → Feedback decoder (64-dim)│  │      │
│  │  │  • Concatenate: [CNN features, SNN feedback] → Attention      │  │      │
│  │  │  • Modulated CNN: CNN features × attention_weights            │  │      │
│  │  └──────────────────────┬───────────────────────────────────────┘  │      │
│  │                         │                                           │      │
│  │                         ▼                                           │      │
│  │  ┌──────────────────────────────────────────────────────────────┐  │      │
│  │  │              Fusion Network                                  │  │      │
│  │  │  Input: [SNN temporal (544-dim), Attended CNN (~12,800-dim)] │  │      │
│  │  │  • FC: fusion_input_size → 768                              │  │      │
│  │  │  • LayerNorm + ReLU + Dropout(0.2)                           │  │      │
│  │  │  • FC: 768 → 512                                             │  │      │
│  │  │  Output: 512-dim unified feature vector                      │  │      │
│  │  └──────────────────────────────────────────────────────────────┘  │      │
│  └───────────────────────────────────────────────────────────────────┘      │
│                                    │                                        │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: CONTEXT-AWARE REWARD LAYER (Dynamic Feedback)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │                  Reward Components (context-aware)             │         │
│  │                                                                │         │
│  │  1. LEVEL REWARDS (Party Level Sum)                            │         │
│  │     • GRIND:    +8.0 per level  ← Highest priority             │         │
│  │     • PROCEED:  +3.0 per level  ← Still valued                 │         │
│  │     • DEFAULT:  +5.0 per level  ← Moderate                     │         │
│  │                                                                │         │
│  │  2. BATTLE WIN REWARDS                                         │         │
│  │     • GRIND:    +4.0 per win   ← Encourage training            │         │
│  │     • PROCEED:  +1.0 per win   ← Low priority                  │         │
│  │     • DEFAULT:  +2.0 per win                                   │         │
│  │                                                                │         │
│  │  3. GRASS TILE REWARDS (Encounter Zones)                       │         │
│  │     • GRIND:    +2.5 per step  ← Seek encounters               │         │
│  │     • PROCEED:  +0.2 per step  ← Avoid battles                 │         │
│  │     • DEFAULT:  +1.0 per step                                  │         │
│  │                                                                │         │
│  │  4. HEALING ITEM USAGE                                         │         │
│  │     • +8.0 per item used (Potion, Super Potion, Revive, etc.)  │         │
│  │     • Encourages resource management                           │         │
│  │                                                                │         │
│  │  5. STRATEGIC GOAL COMPLETION                                  │         │
│  │     • +50.0 when training target reached (GRIND → PROCEED)     │         │
│  │     • Feedback loop: goal achieved → reward → learn            │         │
│  │                                                                │         │
│  │  6. COORDINATE NOVELTY                                         │         │
│  │     • +1.0 per new tile visited (exploration bonus)            │         │
│  │     • Decays with revisits                                     │         │
│  │                                                                │         │
│  │  7. MILESTONE COMPLETION                                       │         │
│  │     • +100.0 per major milestone (gym badge, story event)      │         │
│  │     • +20.0 per minor milestone (reached town, got item)       │         │
│  │                                                                │         │
│  │  8. DIALOG INTERACTION REWARDS                                 │         │
│  │     • +0.1 per A/B press in dialog (minimal reward)            │         │
│  │     • -2.0 dialog farming penalty (after 5 consecutive steps)  │         │
│  │     • Prevents NPC spamming exploitation                       │         │
│  │                                                                │         │
│  │  9. BLACKOUT PENALTY                                           │         │
│  │     • -50.0 for party wipeout                                  │         │
│  │     • Triggers POST_BLACKOUT directive                         │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                    │                                        │
│              Reward = Σ(context_aware_components) - penalties               │
└────────────────────────────────────┼────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PPO POLICY & VALUE NETWORKS                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐          ┌─────────────────────────┐           │
│  │   Policy Network        │          │   Value Network         │           │
│  │   (Actor)               │          │   (Critic)              │           │
│  │                         │          │                         │           │
│  │  512-dim features       │          │  512-dim features       │           │
│  │         ↓               │          │         ↓               │           │
│  │  FC 256 → FC 128        │          │  FC 256 → FC 128        │           │
│  │         ↓               │          │         ↓               │           │
│  │  FC 128 → 6 actions     │          │  FC 128 → 1 value       │           │
│  │                         │          │                         │           │
│  │  Actions:               │          │  V(s) = Expected        │           │
│  │   0. UP                 │          │         cumulative      │           │
│  │   1. DOWN               │          │         reward          │           │
│  │   2. LEFT               │          │                         │           │
│  │   3. RIGHT              │          │                         │           │
│  │   4. A (confirm/talk)   │          │                         │           │
│  │   5. B (cancel/run)     │          │                         │           │
│  └─────────────────────────┘          └─────────────────────────┘           │
│                 │                                                           │
│                 │ Action probabilities π(a|s)                               │
└─────────────────┼───────────────────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ ENVIRONMENT INTERACTION                                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │              ServerAwareEnv (Gymnasium Interface)              │         │
│  │                                                                │         │
│  │  ┌──────────────┐   Action   ┌─────────────────┐               │         │
│  │  │ PPO Agent    │──────────▶ │ mGBA Emulator   │               │         │
│  │  │              │            │ (Pokemon Emerald)│              │         │
│  │  │              │◀────────── │                 │              │         │
│  │  └──────────────┘   State    └─────────────────┘               │         │
│  │                                                                │         │
│  │  ┌────────────────────────────────────────────────┐            │         │
│  │  │        Flask Server (/state API)               │            │         │
│  │  │  • Game state JSON (coordinates, party, etc.)  │            │         │
│  │  │  • Terrain data (15×15 metatile grid)          │            │         │
│  │  │  • Milestone tracking (27 story milestones)    │            │         │
│  │  │  • Server-synced ground truth                  │            │         │
│  │  └────────────────────────────────────────────────┘            │         │
│  │                                                                │         │
│  │  Supporting Modules:                                           │         │
│  │   • SLAM Spatial Mapper (occupancy grid + A* pathfinding)      │         │
│  │   • Milestone Feedback System (prerequisite chains)            │         │
│  │   • Episodic Memory (event tracking with temporal queries)     │         │
│  │   • Battle Engagement Strategy (type-aware decisions)          │         │
│  │   • Healing Item Manager (resource optimization)               │         │
│  │   • Dialogue Processor (NPC interaction + quest tracking)      │         │
│  │   • Terrain Feature Extractor (walkability, behavior encoding) │         │
│  │   • Novelty Detector (coordinate-based exploration rewards)    │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ VISUAL MONITORING DASHBOARD (Real-Time Observability)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              stream.html (1199 lines) + frame_server.py             │    │
│  │                      Real-Time Training Monitor                     │    │
│  │                                                                     │    │
│  │  ┌───────────────────────┐  ┌────────────────────────────────────┐  │    │
│  │  │   Game Screen Panel   │  │    Agent Brain Panel               │  │    │
│  │  │   (895×597 pixels)    │  │    (995px width)                   │  │    │
│  │  │                       │  │                                    │  │    │
│  │  │ • Live GBA frames     │  │ ┌────────────────────────────────┐ │  │    │
│  │  │ • 40 FPS streaming    │  │ │   Agent Thinking (650px)       │ │  │    │
│  │  │ • Frame cache         │  │ │ • VLM reasoning (SSE stream)   │ │  │    │
│  │  │                       │  │ │ • Strategic directives         │ │  │    │
│  │  │ ┌─────────────────┐   │  │ │ • Action decisions             │ │  │    │
│  │  │ │   Milestones    │   │  │ │ • Word-by-word display         │ │  │    │
│  │  │ │   (200px)       │   │  │ └────────────────────────────────┘ │  │    │
│  │  │ │ • 8 recent done │   │  │                                    │  │    │
│  │  │ │ • Color-coded   │   │  │ ┌──────────────┐  ┌──────────────┐ │  │    │
│  │  │ │ • Progress bar  │   │  │ │ Map Visual   │  │ Party Status │ │  │    │
│  │  │ └─────────────────┘   │  │ │ (210px)      │  │ (210px)      │ │  │    │
│  │  │                       │  │ │              │  │              │ │  │    │
│  │  │ ┌─────────────────┐   │  │ │ • ASCII map  │  │ • Pokemon    │ │  │    │
│  │  │ │  Action Queue   │   │  │ │ • NPC (N)    │  │ • Level/HP   │ │  │    │
│  │  │ │  (150px)        │   │  │ │ • Doors (@)  │  │ • Moves/PP   │ │  │    │
│  │  │ │ • 20 recent     │   │  │ │ • Grass (~)  │  │ • HP bars    │ │  │    │
│  │  │ │ • Opacity fade  │   │  │ │ • Player (P) │  │ • Color code │ │  │    │
│  │  │ └─────────────────┘   │  │ └──────────────┘  └──────────────┘ │  │    │
│  │  └───────────────────────┘  └────────────────────────────────────┘  │    │
│  │                                                                     │    │
│  │  Data Flow:                                                         │    │
│  │  • /frame endpoint → Game screen (25ms polling, 40 FPS)             │    │
│  │  • /agent_stream → VLM thinking (Server-Sent Events)                │    │
│  │  • /state → Map + Party (2s polling)                                │    │
│  │  • /milestones → Progress tracking (10s polling)                    │    │
│  │  • /metrics → Cost/tokens/time (2s polling)                         │    │
│  │  • /recent_actions → Button queue (2s polling)                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Purpose: Complete observability for RL training - see what agent sees,     │
│           understand why agent makes decisions, debug in real-time          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│         HELPER TRACKER SYSTEM (State Encoding for SNN/CNN Inputs)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Five specialized tracker classes encode game state into structured vectors │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  1. VLMGuidanceTracker (lines 2170-2200)                       │         │
│  │     Input:  VLM action suggestion (A/B/UP/DOWN/LEFT/RIGHT)     │         │
│  │     Output: 9-dim [one-hot action (7) + confidence + decay]    │         │
│  │     Used:   SNN INPUT 1 (VLM High-Level Goals)                 │         │
│  │                                                                │         │
│  │     Action Map: A=0, B=1, UP=2, DOWN=3, LEFT=4, RIGHT=5, NOOP=6│         │
│  │     Confidence: 1.0 (stuck), 0.8 (normal)                      │         │
│  │     Decay: 50-step window                                      │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  2. SpatialMapTracker (lines 2202-2280)                        │         │
│  │     Input:  Grid search results every 7 VLM calls              │         │
│  │     Output: 11-dim [x, y, explored, directions×4, regions×3, map_id]│    │
│  │     Used:   SNN INPUT 5 (Grid/Map Global Position)             │         │
│  │                                                                │         │
│  │     Directional hints: [north, south, east, west] → Goal       │         │
│  │     Region detection: north/middle/south (Littleroot Town)     │         │
│  │     Explored tiles: Normalized count [0,1]                     │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  3. TerrainFeatureExtractor (lines 2282-2428)                  │         │
│  │     Input:  Server /state API tiles (15×15 metatile grid)      │         │
│  │     Output: 64-dim terrain reasoning vector                    │         │
│  │     Used:   SNN INPUT 6 (Unified Spatial Navigation)           │         │
│  │                                                                │         │
│  │     Features (64 total):                                       │         │
│  │     • Adjacent tiles (4 dirs × 4 features = 16)                │         │
│  │     • Current tile state (8): grass, ledge, water, door, etc.  │         │
│  │     • Movement patterns (8): ledge collisions, grass entries   │         │
│  │     • Padding (32): Reserved for future features               │         │
│  │                                                                │         │
│  │     Ledge logic: ONE-WAY tiles (can't climb up!)               │         │
│  │     Grass logic: WALKABLE tiles (don't avoid!)                 │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  4. DialogueProcessor (lines 2430-2945) ★★★ MULTI-MODAL        │         │
│  │     Input:  NPC dialogue text + location + milestones          │         │
│  │     Output: 64-dim quest encoding + spatial hint flags         │         │
│  │     Used:   SNN INPUT 3 (Structured Quest Memory)              │         │
│  │                                                                │         │
│  │     Parsing Methods:                                           │         │
│  │     • LLM-based: GPT-4 dialogue analysis (optional)            │         │
│  │     • Rule-based: Hard-coded for key NPCs (Birch, May)         │         │
│  │                                                                │         │
│  │     Story Memory (quest tracking):                             │         │
│  │       objective:        "FIND_RIVAL"                           │         │
│  │       target_location:  "ROUTE_103"                            │         │
│  │       target_npc:       "MAY"                                  │         │
│  │       return_npc:       "BIRCH"                                │         │
│  │       linked_milestone: "RECEIVED_POKEDEX"                     │         │
│  │       terrain_hint:     "TALL_GRASS"     ← CNN guidance!       │         │
│  │       direction_hint:   "NORTH"          ← CNN guidance!       │         │
│  │                                                                │         │
│  │     Spatial Hints → CNN Attention:                             │         │
│  │       IF terrain_hint = "TALL_GRASS":                          │         │
│  │           look_for_grass = True                                │         │
│  │           CNN focuses on grass pixels                          │         │
│  │       IF direction_hint = "NORTH":                             │         │
│  │           prefer_direction = NORTH                             │         │
│  │           Policy biased toward UP actions                      │         │
│  │                                                                │         │
│  │     INNOVATION: Language → Visual Attention → RL Policy!       │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  5. MilestoneStateTracker (lines 2948-2978)                    │         │
│  │     Input:  Server /milestones API (27 milestone flags)        │         │
│  │     Output: 16-dim one-hot vector (supports 16 active)         │         │
│  │     Used:   SNN INPUT 2 (Milestone/Story Flags)                │         │
│  │                                                                │         │
│  │     Milestones tracked:                                        │         │
│  │     • RIVAL_BEDROOM, ROUTE_101, STARTER_CHOSEN                 │         │
│  │     • FIRST_RIVAL_BATTLE, RECEIVED_POKEDEX                     │         │
│  │     • DEFEATED_GYM_1, ... (27 total)                           │         │
│  │                                                                │         │
│  │     Prerequisite chains enforced (can't get Pokedex before     │         │
│  │     starter, can't battle gym before reaching town, etc.)      │         │
│  └────────────────────────────────────────────────────────────────┘         │
│                                                                             │
│  Data Flow Summary:                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐               │
│  │   Game   │───▶│ Trackers │───▶│ Encoded  │───▶│   SNN    │              │
│  │  State   │    │ (5 types)│    │ Vectors  │    │  Inputs  │               │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘               │
│      Raw              Process        Structured      Learning               │
│                                                                             │
│  Total Output Dimensions: 9 + 11 + 64 + 64 + 16 = 164 dims                  │
│  (These are distributed across SNN INPUTS 1, 2, 3, 5, 6)                    │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                            INFORMATION FLOW DIAGRAM
═══════════════════════════════════════════════════════════════════════════════

Every 25 steps (VLM Guidance Cycle):
─────────────────────────────────────
  Game Frame → VLM → Strategic Guidance → Directive Update → Context Update
       ↑                                                            │
       └────────────────────────────────────────────────────────────┘

Every step (RL Training Loop):
──────────────────────────────
  State → Feature Extraction → Policy → Action → Environment → Reward
    ↑                                                              │
    └──────────────────────────────────────────────────────────────┘

Feedback Loops:
───────────────
1. GRIND Loop:    Low level → GRIND directive → High grass rewards → Training
                  → Level up → Target reached → PROCEED directive

2. Battle Loop:   Enter battle → Battle state spike (SNN) → Policy adapts
                  → Win battle → +4.0 reward (GRIND) or +1.0 (PROCEED)

3. Healing Loop:  HP < 40% → HEAL directive → Use item → +8.0 reward
                  → HP restored → Resume previous directive

4. Exploration:   New tile → Novelty reward → SLAM updates occupancy grid
                  → A* path planning → Navigate to frontier

5. Milestone:     Complete objective → +100.0 reward → Milestone tracker updates
                  → New target unlocked → VLM receives updated context

═══════════════════════════════════════════════════════════════════════════════
                          KEY ARCHITECTURAL INNOVATIONS
═══════════════════════════════════════════════════════════════════════════════

1. THREE-TIER SEPARATION
   • Strategic (VLM): WHEN to grind, WHEN to progress, WHEN to heal
   • Temporal (SNN): HOW patterns emerge over time (battle rhythm, navigation)
   • Execution (RL): WHICH specific actions maximize context-aware rewards

2. CONTEXT-AWARE REWARDS
   • Same action (grass tile) has different rewards based on strategic context
   • Enables multi-goal RL without separate policies per goal
   • Rewards provide feedback, not rules

3. SNN TEMPORAL ENCODING
   • 9 input streams (544 neurons) encode game state as structured vectors
   • Biologically-inspired temporal reasoning for RL
   • Captures event sequences (dialogue → battle → level up)

4. VLM AS STRATEGIC PLANNER
   • VLM sets goals, not actions (avoids latency issues)
   • Analyzes visual + symbolic state for strategic decisions
   • 4-case battle guidance adapts to party state

5. SLAM-INSPIRED NAVIGATION
   • Occupancy grid built from server terrain data
   • A* pathfinding with landmark detection
   • Exploration bonuses for frontier tiles

═══════════════════════════════════════════════════════════════════════════════
                              MODULE DEPENDENCIES
═══════════════════════════════════════════════════════════════════════════════

Core Training:
  train_rl_vlm_screen_aware.py (3,680 lines)
    ├── agent/rl_trainer.py (PPO + RL config)
    ├── agent/strategic_directive_manager.py (battle strategy)
    ├── agent/battle_engagement_strategy.py (tactical decisions)
    ├── agent/healing_item_manager.py (resource management)
    ├── agent/episodic_memory.py (event tracking)
    ├── agent/vlm_unstuck.py (VLM-based stuck detection)
    ├── utils/snn_encoder.py (SNN temporal encoding)
    ├── utils/vlm.py (multi-backend VLM interface)
    ├── utils/novelty_detector.py (exploration rewards)
    ├── slam_milestone_classes.py (SLAM + milestone tracking)
    └── pokemon_env/emulator.py (mGBA integration)

Server API:
  server/app.py (Flask game state server)
    ├── pokemon_env/memory_reader.py (game state extraction)
    └── pokemon_env/emerald_utils.py (Pokemon-specific data)
  
  server/frame_server.py (FastAPI frame streaming)
    └── .pokeagent_cache/frame_cache.json (shared memory frames)
  
  server/stream.html (1,199 lines - Real-time dashboard)
    ├── /frame endpoint → Game screen display
    ├── /agent_stream → VLM thinking (Server-Sent Events)
    ├── /state → Map + Party visualization
    ├── /milestones → Progress tracking
    └── /recent_actions → Action queue display

SLAM Navigation:
  slam_milestone_classes.py (355 lines)
    ├── SLAMSpatialMapper (occupancy grid + A* pathfinding)
    │   ├── update_position() - Track visited tiles
    │   ├── update_from_server_map() - Ingest 15×15 terrain grid
    │   ├── plan_path() - A* algorithm for navigation
    │   ├── get_exploration_bonus() - Novelty rewards
    │   ├── get_path_following_reward() - On-path progress
    │   └── is_geometrically_consistent() - Map validation
    │
    └── MilestoneFeedbackSystem (27 story milestones)
        ├── check_milestone_completion() - Server sync
        ├── get_prerequisite_chain() - Quest dependencies
        └── get_next_target() - Goal for pathfinding

Utilities:
  utils/map_stitcher.py → utils/map_visualizer.py (spatial visualization)
  utils/pathfinding.py → slam_milestone_classes.py (A* algorithm)
  utils/ocr_dialogue.py → train_rl_vlm_screen_aware.py (text extraction)

═══════════════════════════════════════════════════════════════════════════════
                         SLAM + CNN INTEGRATION DETAILS
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│                  SLAM Occupancy Grid (Global Navigation)                │
│                                                                         │
│   Server provides 15×15 metatile grid with ground-truth terrain:        │
│   ┌──────────────────────────────────────────────────────┐              │
│   │  passable: true/false  (walkability)                 │              │
│   │  behavior: "GRASS", "WATER", "DOOR", "LEDGE", etc.   │              │
│   │  collision: [0,0,0,0] (4 quadrants)                  │              │
│   │  elevation: 3 (for bridge detection)                 │              │
│   └──────────────────────────────────────────────────────┘              │
│                                                                         │
│   SLAM builds occupancy grid from this data:                            │
│   • -1 = Unknown (not visited yet)                                      │
│   •  0 = Free (walkable tile)                                           │
│   •  1 = Blocked (wall, NPC, obstacle)                                  │
│   •  2 = Landmark (Gym, Pokemon Center, Lab, NPC)                       │
│                                                                         │
│   Example occupancy grid (Littleroot Town):                             │
│   ┌─────────────────────────────────────────────┐                       │
│   │  1  1  1  1  1  1  1  1  1  1  1  1  1  1   │                       │
│   │  1  2  0  0  0  0  0  0  0  0  0  2  1  1   │  2 = Landmarks        │
│   │  1  0  0  0  0  0  0  0  0  0  0  0  1  1   │  (Lab + Rival house)  │
│   │  1  0  0  0  0  0  0  P  0  0  0  0  1  1   │  P = Player           │
│   │  1  1  1  0  0  0  0  0  0  0  1  1  1  1   │  1 = Walls            │
│   │  1  1  1  1  1  0  0  0  1  1  1  1  1  1   │  0 = Walkable         │
│   └─────────────────────────────────────────────┘                       │
│                                                                         │
│   A* Pathfinding uses this grid:                                        │
│   • Start: Player position (map_id, x, y)                               │
│   • Goal: Milestone target (e.g., Rustboro Gym entrance)                │
│   • Cost: Manhattan distance + occupancy penalties                      │
│   • Output: List of waypoints [(x1,y1), (x2,y2), ...]                   │
│                                                                         │
│   Exploration Tracking:                                                 │
│   • First visit to tile: +1.0 novelty reward                            │
│   • Second visit: +0.5 reward                                           │
│   • Third visit: +0.25 reward (exponential decay)                       │
│   • Encourages frontier exploration                                     │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│              CNN Spatial Features (Local Vision)                        │
│                                                                         │
│   CNN sees 84×84 pixel game screen (4-frame stack):                     │
│   ┌────────────────────────────────────────────┐                        │
│   │  Input: (4, 84, 84) tensor                 │                        │
│   │    Frame 1: Current game screen            │                        │
│   │    Frame 2: t-1 (motion detection)         │                        │
│   │    Frame 3: t-2                            │                        │
│   │    Frame 4: t-3 (temporal context)         │                        │
│   └────────────────────────────────────────────┘                        │
│                                                                         │
│   CNN Architecture (4 conv layers):                                     │
│   Layer 1: 32 filters, 8×8 kernel → 21×21 feature maps                  │
│     Purpose: Detect edges, basic shapes                                 │
│                                                                         │
│   Layer 2: 64 filters, 4×4 kernel → 10×10 feature maps                  │
│     Purpose: Detect objects (doors, NPCs, grass patches)                │
│                                                                         │
│   Layer 3: 128 filters, 3×3 kernel → 10×10 feature maps                 │
│     Purpose: Detect spatial relationships (NPC blocking door)           │
│                                                                         │
│   Layer 4: 128 filters, 3×3 kernel → 10×10×128 = 12,800 features        │
│     Purpose: High-level scene understanding                             │
│                                                                         │
│   FC Layer: 12,800 → 256 spatial features                               │
│                                                                         │
│   Output: 256-dim vector encoding:                                      │
│   • Door locations (relative to player)                                 │
│   • NPC positions and orientations                                      │
│   • Terrain types (grass, water, sand)                                  │
│   • Visual cues (signposts, buildings)                                  │
│   • Motion patterns (NPC walking, grass animation)                      │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│           CNN + SLAM Complementarity (Local + Global)                   │
│                                                                         │
│   CNN (Local):                    SLAM (Global):                        │
│   ────────────                    ─────────────                         │
│   • "I see a door"                • "That door leads to Route 101"      │
│   • "NPC is 3 tiles away"         • "Route 101 is on my path to gym"    │
│   • "Grass patch ahead"           • "I've visited this area 5 times"    │
│   • "Building entrance nearby"    • "This is Oldale Pokemon Center"     │
│   • Visual appearance             • Semantic meaning + history          │
│                                                                         │
│   Example Decision Flow:                                                │
│   1. SLAM: "Navigate to Rustboro Gym (20 tiles north)"                  │
│   2. A* Path: [(x,y), (x,y+1), (x,y+2), ... gym entrance]               │
│   3. CNN: "I see a door at (x, y+1) - that's on my path"                │
│   4. Policy: "Press UP to reach door, then A to enter"                  │
│   5. CNN: "Screen changed - map transition detected"                    │
│   6. SLAM: "Update occupancy grid, moved to new map_id"                 │
│                                                                         │
│   Both feed into 512-dim feature vector:                                │
│   [CNN spatial (256) | SNN temporal (5) | Milestones (16) | ...]        │
│                       ↓                                                 │
│                  PPO Policy Network                                     │
│                       ↓                                                 │
│                 Action: UP/DOWN/LEFT/RIGHT/A/B                          │
└─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
                    COORDINATE SYSTEM ARCHITECTURE ★★★
═══════════════════════════════════════════════════════════════════════════════

The coordinate system is the spatial foundation that connects all subsystems.
It provides a unified (map_id, x, y) reference frame flowing through the entire
pipeline, enabling multi-scale spatial reasoning from local vision to global
pathfinding to strategic planning.

┌─────────────────────────────────────────────────────────────────────────┐
│                  COORDINATE SYSTEM DATA FLOW (8 Stages)                 │
└─────────────────────────────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════╗
║ STAGE 1: SERVER STATE (Ground Truth)                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
    Flask API: GET http://localhost:8000/state
    Returns:
    {
      "player": {
        "map_id": 37,           ← Integer (0-255), e.g., 37 = Route 101
        "x": 7,                 ← Tile X coordinate (0-based)
        "y": 5,                 ← Tile Y coordinate (0-based)
        "location": "ROUTE 101"
      },
      "tiles": [15×15 array]    ← Metatile grid centered on player
    }
    
    Coordinate Space Properties:
    • Map ID: 256 possible maps (Littleroot Town = 4, Route 101 = 37)
    • Tile Coords: Per-map system (not global), range 0-255 per axis
    • Origin: (0, 0) = top-left corner of each map
    
             ↓ (map_id, x, y, tiles)

╔═══════════════════════════════════════════════════════════════════════════╗
║ STAGE 2: SPATIAL MAP TRACKER (Position Encoding)                          ║
╚═══════════════════════════════════════════════════════════════════════════╝
    SpatialMapTracker.encode(x=7, y=5, map_id=37)
    Produces 11-dimensional vector:
    
    [0.028, 0.020, 2.0, 0.8, 0.3, 0.1, 0.2, 1.0, 0.0, 0.0, 0.145]
      │      │     │    │    │    │    │    │    │    │      │
      │      │     │    │    │    │    │    │    │    │      └─ map_id_norm
      │      │     │    │    │    │    │    │    │    └──────── field region
      │      │     │    │    │    │    │    │    └───────────── town region
      │      │     │    │    │    │    │    └────────────────── outdoor region
      │      │     │    │    │    │    └─────────────────────── W exploration
      │      │     │    │    │    └──────────────────────────── E exploration
      │      │     │    │    └───────────────────────────────── S exploration
      │      │     │    └────────────────────────────────────── N exploration
      │      │     └─────────────────────────────────────────── explored_count
      │      └───────────────────────────────────────────────── y_norm (0-1)
      └──────────────────────────────────────────────────────── x_norm (0-1)
    
    Grid Search Integration (every 7 VLM calls):
    • Explores 8 surrounding tiles: (x±1, y±1)
    • Updates exploration_count[(map_id, x, y)]
    • Feeds results to VLM for spatial awareness
    
             ↓ 11-dim position vector

╔═══════════════════════════════════════════════════════════════════════════╗
║ STAGE 3: SNN INPUT 5 (Neural Position Encoding)                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
    11-dimensional vector → 60 spiking neurons
    
    Spike Encoding:
    • x_norm=0.028 → Neurons 1-5 fire at 2.8% intensity
    • explored_count=2.0 → Neurons 15-20 fire moderately
    • map_id_norm=0.145 → Neurons 55-60 encode map context
    
    Purpose: Agent "feels" its spatial position as neural activity
    
             ↓ Spatio-temporal spike trains

╔═══════════════════════════════════════════════════════════════════════════╗
║ STAGE 4: CNN SPATIAL CONTEXT (Local Vision)                               ║
╚═══════════════════════════════════════════════════════════════════════════╝
    15×15 metatile grid → 84×84 pixel screen → ResNet18
    
    ┌─────────────────────────────────────┐
    │ [G][G][G][G][G][G][G][G][G][G][G] │  G = Grass
    │ [G][G][G][P][G][G][G][G][G][G][G] │  P = Player (7,5)
    │ [G][G][G][G][N][G][G][G][G][G][G] │  N = NPC
    │ [G][G][G][G][G][G][G][D][G][G][G] │  D = Door
    │ [═][═][═][═][═][═][═][═][═][═][═] │  ═ = Ledge
    └─────────────────────────────────────┘
    
    CNN extracts 512-dim features:
    • Door location: 3 tiles right, 1 tile down
    • NPC position: 1 tile right
    • Grass density: 90%
    • Ledge boundary: 2 tiles south
    
             ↓ 512-dim local spatial features → SNN INPUT 1 (256 neurons)

╔═══════════════════════════════════════════════════════════════════════════╗
║ STAGE 5: SLAM OCCUPANCY GRID (Global Map)                                 ║
╚═══════════════════════════════════════════════════════════════════════════╝
    slam_mapper.update_position(map_id=37, x=7, y=5, tiles=...)
    
    Builds 100×100 occupancy grid per map:
    • 0.0 = Unexplored
    • 1.0 = Explored, walkable
    • 0.5 = Obstacle (wall, NPC)
    
    Route 101 Occupancy Grid Example:
    ┌──────────────────────────────────────┐
    │ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  │ ← Unexplored north
    │ 0.0  1.0  1.0  1.0  1.0  0.0  0.0  │ ← Explored path
    │ 0.0  1.0  0.5  1.0  P    1.0  0.0  │ ← Player at (7,5)
    │ 0.0  1.0  1.0  1.0  1.0  1.0  0.0  │ ← Known walkable
    │ 0.0  0.0  0.0  0.0  0.0  0.0  0.0  │ ← Unexplored south
    └──────────────────────────────────────┘
    
    A* Pathfinding:
    path = slam_mapper.find_path(
        start=(map_id=37, x=7, y=5),     ← Current position
        goal=(map_id=37, x=7, y=50)      ← Route 103 entrance
    )
    → Returns waypoint list: [(7,6), (7,7), (7,8), ..., (7,50)]
    
             ↓ Exploration bonus + path following reward

╔═══════════════════════════════════════════════════════════════════════════╗
║ STAGE 6: VLM LOCATION-AWARE GUIDANCE                                      ║
╚═══════════════════════════════════════════════════════════════════════════╝
    VLM Prompt with Coordinate Context:
    ┌────────────────────────────────────────────────────────────┐
    │ You are guiding a Pokemon Emerald speedrun agent.          │
    │                                                            │
    │ Current State:                                             │
    │   Location: Route 101                                      │
    │   Position: Map 37 at (7, 5)                               │
    │   Explored: 2 times (moderate familiarity)                 │
    │                                                            │
    │ Grid Search Results:                                       │
    │   North (7,4): Tall grass with wild Pokemon                │
    │   South (7,6): Path continues toward Route 103             │
    │   East (8,5): NPC Trainer looking for battles              │
    │   West (6,5): Tall grass (previously explored)             │
    │                                                            │
    │ Mission: Reach Professor Birch at Route 103 entrance       │
    │                                                            │
    │ Provide next waypoint with reasoning.                      │
    └────────────────────────────────────────────────────────────┘
    
    VLM Response:
    {
      "waypoint": "Move south toward Route 103 entrance at (7, 50)",
      "reasoning": "Avoid trainer battle (east), take direct path south",
      "action_hint": "DOWN"
    }
    
             ↓ Coordinate-grounded strategic guidance

╔═══════════════════════════════════════════════════════════════════════════╗
║ STAGE 7: COORDINATE NOVELTY REWARDS                                       ║
╚═══════════════════════════════════════════════════════════════════════════╝
    Exploration Tracking:
    exploration_count[(map_id, x, y)] → visit count
    
    Reward Calculation:
    if (37, 7, 5) not in exploration_count:
        reward += 3.0                    # First visit bonus! ★
        exploration_count[(37, 7, 5)] = 1
    elif exploration_count[(37, 7, 5)] == 1:
        reward += 1.0                    # Second visit
    elif exploration_count[(37, 7, 5)] == 2:
        reward += 0.3                    # Third visit (diminishing)
    
    SLAM Exploration Bonus:
    slam_bonus = slam_mapper.get_exploration_bonus(map_id=37, x=7, y=5)
    # → +2.0 if expanding frontier, +0.5 if revisiting known area
    
    Path Following Reward:
    if moving toward VLM waypoint:
        path_reward += 2.0               # On correct path
    else:
        path_reward -= 0.5               # Deviating from plan
    
             ↓ Spatial rewards → PPO policy gradient

╔═══════════════════════════════════════════════════════════════════════════╗
║ STAGE 8: TERRAIN & DIALOGUE SPATIAL FEATURES                              ║
╚═══════════════════════════════════════════════════════════════════════════╝
    TerrainFeatureExtractor uses position to:
    • Classify tile type at (7, 5) → "GRASS"
    • Detect adjacent tiles: N=grass, S=grass, E=NPC, W=grass
    • Encode as 64-dim vector → SNN INPUT 6 (60 neurons)
    
    DialogueProcessor spatial hints:
    • NPC dialogue: "Head north to Route 103!" → Directional vector
    • Signpost: "Oldale Town - East" → Target coordinates
    • Encode as 64-dim vector → SNN INPUT 7 (60 neurons)
    
             ↓ All spatial information converges in SNN

┌─────────────────────────────────────────────────────────────────────────┐
│                    FINAL INTEGRATION: SNN → PPO → ACTION                │
│                                                                         │
│   Active SNN Streams (coordinate-informed):                             │
│   • VLM goals, milestones, quest memory                                 │
│   • Grid/map tracker + terrain encoder + RGB analyzer                   │
│   • SLAM hints + k-NN recall (confidence + spike flag)                  │
│                                                                         │
│   SNN processes 10 temporal steps (100ms window):                       │
│   • Spike patterns encode spatial-temporal reasoning                    │
│   • 544 neurons → 256-dim readout vector                                │
│                                                                         │
│   PPO Policy receives:                                                  │
│   [SNN features (256) | CNN features (512) | ... ] → 1024-dim state     │
│                       ↓                                                 │
│   Actor head: π(a|s) → Action probabilities [UP, DOWN, LEFT, RIGHT, A, B]│
│                       ↓                                                 │
│   Environment executes action → New (map_id, x, y) → Cycle repeats      │
└─────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
         COORDINATE SYSTEM INNOVATION SUMMARY ★★★
═══════════════════════════════════════════════════════════════════════════════

Multi-Scale Spatial Reasoning:
┌──────────────────────────────────────────────────────────────────────────┐
│ Scale         │ Component      │ Resolution  │ Purpose                   │
├───────────────┼────────────────┼─────────────┼───────────────────────────┤
│ MICRO         │ CNN            │ 15×15 tiles │ Obstacle avoidance        │
│ MESO          │ SpatialTracker │ (x,y) coord │ Navigation context        │
│ MACRO         │ SLAM           │ 100×100 map │ Long-term pathfinding     │
│ META          │ VLM            │ Natural lang│ Strategic waypoint setting│
└──────────────────────────────────────────────────────────────────────────┘

Novel Integration:
• Coordinates = "spatial glue" connecting reactive (CNN) + deliberative (SLAM)
                + strategic (VLM) layers
• Grid search: Bidirectional feedback (coordinates → VLM → refined coordinates)
• Coordinate novelty: Exploration without explicit map-drawing supervision

Publication Angle:
"Hierarchical Spatial Reasoning via Coordinate-Grounded Multi-Modal Learning"

═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────────┐
│             MULTI-LAYERED REWARD SYSTEM (13 Components)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Reward Layer 1: MILESTONE REWARDS (+100 major, +20 minor, +80 story)       │
│  Reward Layer 2: RGB PIXEL TERRAIN (tree/grass detection, walkability)      │
│  Reward Layer 3: KNN VISUAL NOVELTY (+2.0 novel frames, L2 distance >1000)  │
│  Reward Layer 4: COORDINATE EXPLORATION (+3.0 first visit, SLAM bonuses)    │
│  Reward Layer 5: MAP TRANSITIONS (+20 new map, +30 brand new)               │
│  Reward Layer 6: SLAM PATH FOLLOWING (+2.0 on path, -1.0 lost)              │
│  Reward Layer 7: CONTEXT BATTLE/LEVEL (8.0× GRIND, 3.0× PROCEED)            │
│  Reward Layer 8: TERRAIN GRASS (+2.5 GRIND, +0.2 PROCEED)                   │
│  Reward Layer 9: HEALING ITEMS (+8.0 per usage)                             │
│  Reward Layer 10: STRATEGIC GOALS (+50 training complete)                   │
│  Reward Layer 11: BLACKOUT PENALTY (-50 party wipeout)                      │
│  Reward Layer 12: VLM GUIDANCE (+5.0 stuck escape)                          │
│  Reward Layer 13: FEEDBACK LOOPS (see below)                                │
│                                                                             │
│  TOTAL: 13 components, context-aware, hierarchically organized              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  FEEDBACK LOOP 1: NOVELTY → SNN PRIORITY                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  KNN Frame Comparison (line 1484)                                           │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ min_dist = min([norm(current-past) for past in buffer])       │          │
│  │ if min_dist > 1000:                                           │          │
│  │     novelty_reward = +2.0                                     │          │
│  │     milestone_priority_signal += 0.1  # ★ FEEDBACK            │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                           │                                                 │
│                           ▼                                                 │
│  Feed into SNN INPUT 9 (line 3457)                                          │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ battle_encoding = cat([badge_count,                           │          │
│  │                        milestone_priority_signal,  # ← Here   │          │
│  │                        battle_state])  # 32-dim → SNN         │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                           │                                                 │
│                           ▼                                                 │
│  SNN Processing: High priority → increased spike rate → "keep exploring!"   │
│                                                                             │
│  RESULT: Visual novelty boosts exploration behavior                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  FEEDBACK LOOP 2: SNN → CNN ATTENTION                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SNN Temporal Features (544-dim: mean of 10-timestep spike history)       │
│                           │                                                 │
│                           ▼                                                 │
│  Feedback Decoder (line 3930-3935)                                          │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ snn_feedback = snn_feedback_decoder(snn_temporal_features)    │          │
│  │ # 544-dim → 128 → 64-dim navigation hints                     │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                           │                                                 │
│                           ▼                                                 │
│  Concatenate with CNN (line 4324)                                           │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ attention_input = cat([cnn_features(~12,800), snn_feedback(64)])│          │
│  │ # CNN flattened conv output + SNN navigation hints            │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                           │                                                 │
│                           ▼                                                 │
│  Attention Modulation (line 4325-4326)                                      │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ spatial_weights = spatial_attention(attention_input)          │          │
│  │ attended_cnn = cnn_features * spatial_weights  # Modulate   │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                           │                                                 │
│                           ▼                                                 │
│  Fused Features (line 4334-4337):                                           │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ fused = cat([snn_temporal_features(544), attended_cnn(~12,800)])│          │
│  │ # Total: ~13,408-dim → Fusion network → 512-dim               │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  RESULT: SNN temporal reasoning guides CNN spatial attention                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  FEEDBACK LOOP 3: MILESTONE → SNN PRIORITY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Milestone Completion                                                       │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ if milestone_achieved:                                        │          │
│  │     reward = +100.0                                           │          │
│  │     milestone_priority_signal = 1.0  # ★ Max priority         │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                           │                                                 │
│                           ▼                                                 │
│  SNN INPUT 9 (line 1147, 3457)                                              │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ battle_encoding = cat([badge_count(16),                       │          │
│  │                        milestone_priority_signal(1),  # ← Here│          │
│  │                        battle_state(15)])  # 32-dim           │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                           │                                                 │
│                           ▼                                                 │
│  SNN Encodes Importance: High priority → "Remember this state pattern!"     │
│                                                                             │
│  RESULT: Milestones strengthen memory of successful action sequences        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  FEEDBACK LOOP 4: VLM → REWARD MODULATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  VLM Analysis                                                               │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ Input: frame, party_levels, milestone, episodic_memory        │          │
│  │ VLM: "Party level 5, need level 10 for Rival → GRIND"         │          │
│  │ Output: directive = "GRIND"                                   │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                           │                                                 │
│                           ▼                                                 │
│  Reward Modulation                                                          │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ if directive == "GRIND":                                      │          │
│  │     level_mult = 8.0    # Highest priority                    │          │
│  │     grass_mult = 2.5    # Seek encounters                     │          │
│  │     battle_bonus = 4.0  # Encourage fights                    │          │
│  │ elif directive == "PROCEED":                                  │          │
│  │     level_mult = 3.0    # Lower priority                      │          │
│  │     grass_mult = 0.2    # Avoid encounters                    │          │
│  │     battle_bonus = 1.0  # Discourage fights                   │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                           │                                                 │
│                           ▼                                                 │
│  Policy Learning                                                            │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ GRIND mode: "walk in grass = +2.5 reward = good"              │          │
│  │ PROCEED mode: "walk in grass = +0.2 reward = bad"             │          │
│  │ Same action, different context → context-aware strategy       │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                           │                                                 │
│                           ▼                                                 │
│  Directive Transition                                                       │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ if party_level >= target_level:                               │          │
│  │     directive = "PROCEED"  # Goal complete!                   │          │
│  │     goal_reward = +50.0                                       │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  RESULT: High-level strategy modulates low-level learning signals           │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  COMPLETE FEEDBACK CYCLE                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│      VLM (Strategy)                                                         │
│           │                                                                 │
│           │ Directives                                                        │
│           ▼                                                                   │
│      Rewards (Tactics) ◀──── Novelty/Milestones (Priority Signals)           │
│           │                                                                   │
│           │ Learning Signals                                                  │
│           ▼                                                                   │
│      SNN (Temporal) ───────────────┐                                          │
│           │                        │ Attention Guidance                       │
│           │ Priority Encoding      ▼                                          │
│           ▼                    CNN (Spatial)                                  │
│      PPO Policy (Execution)        │                                          │
│           │                        │                                          │
│           └────────────────────────┘                                          │
│                     │                                                         │
│                     │ Actions                                                 │
│                     ▼                                                         │
│              Environment (Game)                                               │
│                     │                                                         │
│                     │ New State                                               │
│                     ▼                                                         │
│              [CYCLE REPEATS]                                                  │
│                                                                               │
│  KEY INNOVATIONS:                                                             │
│  • Multi-level feedback: Rewards → SNN → CNN → Policy                        │
│  • Context propagation: VLM → Rewards → Policy learning                      │
│  • Temporal guidance: SNN → CNN attention modulation                         │
│  • Priority signaling: Novelty/Milestones → SNN exploration bias             │
│  • Hierarchical learning: Strategy ↔ Tactics ↔ Execution                     │
│  • Pixel-level understanding: RGB analysis complements server data           │
│  • Visual novelty: KNN frame comparison rewards novel experiences            │
│  • Milestone stagnation detection: Tracks time/steps since last milestone    │
│  • Server location tracking: Uses map_id, x, y for comprehensive stuck detection│
│  • VLM suggestion pattern tracking: Detects repetitive guidance              │
└───────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  ANTI-EXPLOITATION: DIALOG FARMING PREVENTION ★★★ NEW                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Problem Discovered (Nov 23, 2024):                                         │
│  Agent found it could earn constant +0.5 reward by pressing A in dialog     │
│  state. Littleroot Town has many NPCs, so agent wandered town spamming     │
│  NPCs for easy rewards instead of progressing to Route 103.                 │
│                                                                             │
│  Why Dialog Farming Occurred:                                               │
│  • Dialog reward (+0.5) was constant and immediate                          │
│  • VLM guidance (+8.0) only given when exact button matched                 │
│  • Exploration (+3.2) required finding new tiles                            │
│  • Dialog farming: just walk to nearest NPC, press A, repeat                │
│  • Agent optimized for "easy constant reward" over "hard strategic reward"  │
│                                                                             │
│  Solution Implemented:                                                      │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ Component: Dialog Farming Penalty System                      │          │
│  │ File: train_rl_vlm_screen_aware.py (lines 1504-1527)          │          │
│  │                                                               │          │
│  │ State Tracking:                                               │          │
│  │   consecutive_dialog_steps = 0  # Counter                     │          │
│  │                                                               │          │
│  │ Logic:                                                        │          │
│  │   if game_state == 'dialog':                                  │          │
│  │       consecutive_dialog_steps += 1                           │          │
│  │                                                               │          │
│  │       if action in ['A', 'B']:                                │          │
│  │           if consecutive_dialog_steps > 5:                    │          │
│  │               reward -= 2.0  # Heavy penalty!                 │          │
│  │           else:                                               │          │
│  │               reward += 0.1  # Minimal (was 0.5)              │          │
│  │   else:                                                       │          │
│  │       consecutive_dialog_steps = 0  # Reset                   │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  Key Features:                                                              │
│  • Threshold-based: First 5 dialog steps still rewarded (+0.1)              │
│  • Heavy penalty: After 5 consecutive steps, -2.0 per A/B press             │
│  • Automatic reset: Counter resets when agent leaves dialog state           │
│  • Net effect: Extended dialog farming becomes unprofitable                 │
│    - Normal dialog: 5 steps × +0.1 = +0.5 total (OK)                        │
│    - Farming attempt: 10 steps = 5×(+0.1) + 5×(-2.0) = -9.5 (BAD!)         │
│                                                                             │
│  Reward Hierarchy Enforcement:                                              │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ Milestone (+100.0)           ← Story progress (highest)       │          │
│  │   └─ VLM Alignment (+8.0)        ← Strategic guidance         │          │
│  │       └─ Map Transition (+20.0)      ← New areas              │          │
│  │           └─ New Tile (+3.2)             ← Exploration        │          │
│  │               └─ Dialog (+0.1)               ← Minimal        │          │
│  │                   └─ Dialog Farming (-2.0)       ← Penalty    │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  Design Rationale:                                                          │
│  1. Small constant rewards create exploitation opportunities                │
│  2. Threshold allows legitimate dialog interactions                         │
│  3. Heavy penalty makes farming strategy dominated by alternatives          │
│  4. Automatic reset prevents false positives (menus, battle text)           │
│                                                                             │
│  Training Impact:                                                           │
│  • Before fix: Agent stuck in Littleroot for 5300+ steps                    │
│  • After fix: Agent follows VLM guidance, progresses to Route 103           │
│  • Logged: "❌ Dialog farming penalty -2.0" when triggered                  │
│                                                                             │
│  Publication Significance:                                                  │
│  This is a novel anti-exploitation mechanism for RL agents in games with    │
│  high-frequency interaction opportunities. Shows importance of:             │
│  • Threshold-based penalties vs blanket restrictions                        │
│  • Reward hierarchy design (strategic > tactical > constant)                │
│  • Context-aware reward shaping (dialog OK, dialog farming BAD)             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  SEQUENTIAL PHASE MATCHING ★★★ MODIFIED                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Problem Discovered (Nov 22-23, 2024):                                      │
│  Story context (_get_story_context) and action guidance (_get_milestone     │
│  _action) used different logic, causing misalignment:                       │
│  • Story: "Battle rival on Route 103 to get Pokedex"                        │
│  • Action: "Talk to Birch for Pokedex" (WRONG - rival not battled yet!)    │
│                                                                             │
│  Root Cause:                                                                │
│  _get_story_context checked FIRST_RIVAL_BATTLE milestone before showing     │
│  "return to lab" story, but _get_milestone_action didn't check it, causing  │
│  agent to receive conflicting guidance and get stuck at lab.                │
│                                                                             │
│  Solution Implemented:                                                      │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ Refactored _get_milestone_action() (lines 235-268)            │          │
│  │ To match _get_story_context() logic EXACTLY                   │          │
│  │                                                               │          │
│  │ Sequential Phase Structure:                                   │          │
│  │                                                               │          │
│  │ Phase 1: Rival Bedroom (prerequisite for everything)          │          │
│  │   if not has_rival_bedroom and not has_pokemon:               │          │
│  │       return "Find rival's house, go UP/NORTH..."             │          │
│  │                                                               │          │
│  │ Phase 2: Route 101 (unlock world access)                      │          │
│  │   elif not has_route_101 and not has_pokemon:                 │          │
│  │       return "Exit town, go UP/NORTH to Route 101..."         │          │
│  │                                                               │          │
│  │ Phase 3: Get Starter Pokemon (core quest)                     │          │
│  │   elif not has_starter and not has_pokemon:                   │          │
│  │       return "Find Birch, pick starter Pokemon..."            │          │
│  │                                                               │          │
│  │ Phase 4: Pokedex Quest (COMPLEX - with sub-phases)            │          │
│  │   elif has_pokemon and not has_pokedex:                       │          │
│  │       if 'FIRST_RIVAL_BATTLE' in completed:                   │          │
│  │           # Sub-phase 4a: Return to lab                       │          │
│  │           if at_lab:                                          │          │
│  │               return "Talk to Birch for Pokedex..."           │          │
│  │           else:                                               │          │
│  │               return "Return to Birch's Lab..."               │          │
│  │       else:                                                   │          │
│  │           # Sub-phase 4b: Find rival first                    │          │
│  │           if at_lab:                                          │          │
│  │               return "Exit lab, go to Route 103..."           │          │
│  │           elif "ROUTE_103" in location:                       │          │
│  │               return "Look for rival, battle them..."         │          │
│  │           else:                                               │          │
│  │               return "Go north to Route 103..."               │          │
│  │                                                               │          │
│  │ Phase 5: Post-Pokedex (open world)                            │          │
│  │   elif has_pokedex:                                           │          │
│  │       return "Explore! Head to Petalburg City..."             │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  Key Feature: Sub-Phase Gating                                              │
│  Phase 4 has TWO sub-phases with milestone gating:                          │
│  • Must complete FIRST_RIVAL_BATTLE before "return to lab" guidance         │
│  • Agent cannot skip quest steps (battle rival → get Pokedex)               │
│  • Both story context and action guidance check the same milestones         │
│                                                                             │
│  Architecture Impact:                                                       │
│  • Hierarchical quest progression: Linear phases → Gated sub-phases         │
│  • Consistency guarantee: Story and action use identical phase logic        │
│  • Prerequisite enforcement: Cannot progress without completing prior steps │
│  • Clear milestone dependencies: FIRST_RIVAL_BATTLE gates Pokedex quest     │
│                                                                             │
│  Training Impact:                                                           │
│  • Before fix: Agent stuck at lab with Pokemon, no Pokedex (24+ hours)      │
│  • After fix: Agent follows correct sequence (Route 103 → Battle → Lab)     │
│  • VLM receives aligned guidance, no conflicting instructions               │
│                                                                             │
│  Publication Significance:                                                  │
│  Demonstrates importance of hierarchical goal alignment in multi-tier RL    │
│  systems. Shows that high-level (VLM) and mid-level (milestone) guidance    │
│  must use consistent logic to avoid agent confusion and stuck states.       │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  VLM CHARACTER IDENTITY CLARIFICATION ★★ MODIFIED                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Problem Discovered (Nov 23, 2024):                                         │
│  VLM hallucinated player character identity:                                │
│  • VLM output: "Birch is standing in front of the Pokemon Lab..."           │
│  • Reality: Player (Brendan/May) is standing, Birch is the professor        │
│  • Confusion between player sprite and Professor Birch character            │
│                                                                             │
│  Root Cause:                                                                │
│  VLM prompt lacked explicit character identity clarification. Prompt asked  │
│  for "visual description" without specifying that player ≠ Birch.           │
│                                                                             │
│  Solution Implemented:                                                      │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ Enhanced VLM Prompt (lines 814-820, 847-851)                  │          │
│  │                                                               │          │
│  │ Change 1: Character Identity Header                           │          │
│  │   """Pokemon Emerald Speedrun - Story Progress                │          │
│  │                                                               │          │
│  │   🎮 **YOU ARE THE PLAYER** (Brendan/May), a young            │          │
│  │      Pokemon trainer starting your journey!                   │          │
│  │   📚 **Professor Birch** is a Pokemon researcher who          │          │
│  │      gives you your first Pokemon and Pokedex                 │          │
│  │   👥 **Your Rival** (May/Brendan) is Birch's child            │          │
│  │      who also becomes a trainer                               │          │
│  │   """                                                         │          │
│  │                                                               │          │
│  │ Change 2: Anti-Hallucination Rule                             │          │
│  │   **CRITICAL RULES:**                                         │          │
│  │   1. You MUST start with "LOCATION:"                          │          │
│  │   2. You MUST include "ACTION:" with ONE button               │          │
│  │   3. Do NOT write paragraphs outside this format              │          │
│  │   4. DO NOT describe the player as "Birch" - YOU are          │          │
│  │      the player, Birch is the professor!                      │          │
│  │   5. Your goal: {story_context['goal']}                       │          │
│  │                                                               │          │
│  │ Change 3: Description Scope                                   │          │
│  │   Before: "Provide a brief visual description"                │          │
│  │   After: "Describe **your surroundings**, NOT player"         │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  Verification:                                                              │
│  • Before fix: "LOCATION: LITTLEROOT_TOWN - Birch is standing..."           │
│  • After fix: "LOCATION: LITTLEROOT_TOWN - The player is in a small town..."│
│  • ✅ VLM now correctly describes surroundings, not player character         │
│                                                                             │
│  Architecture Impact:                                                       │
│  • Prompt engineering layer: Explicit character role assignment             │
│  • Anti-hallucination system: Rule-based constraints on VLM output          │
│  • Scope clarification: Distinguish player description from environment     │
│                                                                             │
│  Training Impact:                                                           │
│  • Improved VLM accuracy: No more character confusion                       │
│  • Better location descriptions: Focus on actionable environment features   │
│  • Consistent character references: Player/Birch/Rival correctly identified │
│                                                                             │
│  Publication Significance:                                                  │
│  Highlights importance of explicit role assignment in VLM prompts for       │
│  game-playing agents. Shows that even capable models need clear identity    │
│  grounding to avoid hallucination in multi-character scenarios.             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  MILESTONE STAGNATION DETECTION ★★★ NEW (Nov 24, 2024)                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Problem Discovered:                                                        │
│  Agent gets stuck after visiting Birch's Lab (BIRCH_LAB_VISITED milestone)  │
│  and doesn't progress for hours, with VLM repeatedly suggesting same actions│
│                                                                             │
│  Solution Implemented:                                                      │
│  ┌───────────────────────────────────────────────────────────────┐          │
│  │ Component: Comprehensive Stagnation Detection                 │          │
│  │ File: train_rl_vlm_screen_aware.py (lines 1899-1947)          │          │
│  │                                                               │          │
│  │ Detection Methods (OR logic - any triggers stagnation):      │          │
│  │   1. Time-based: >30 minutes without new milestone            │          │
│  │   2. Step-based: >10,000 steps without milestone progress     │          │
│  │   3. Location-based: Same area for >5,000 steps                │          │
│  │   4. VLM-based: Only 1-2 unique suggestions in last 100 calls│          │
│  │                                                               │          │
│  │ Data Sources Used:                                            │          │
│  │   • Server milestone API (/milestones endpoint)              │          │
│  │   • Server location data (map_id, x, y, location_name)       │          │
│  │   • VLM suggestion history (last 200 suggestions)           │          │
│  │   • Position history (last 12 positions for loop detection)  │          │
│  │                                                               │          │
│  │ When Stagnation Detected:                                     │          │
│  │   • Moderate penalty: -5.0 reward                             │          │
│  │   • Enhanced VLM prompt: Includes stagnation warnings         │          │
│  │   • Alternative actions: VLM tries different directions       │          │
│  │   • More frequent VLM calls: Every 20 steps (vs 50 normal)    │          │
│  │                                                               │          │
│  │ VLM Prompt Enhancement:                                       │          │
│  │   "⚠️ MILESTONE STAGNATION DETECTED!                           │          │
│  │    No new milestones for {steps} steps!                       │          │
│  │    Server Location: {location} at {position}                 │          │
│  │    Recent locations: {unique_count} unique areas             │          │
│  │    VLM Suggestions: {unique_actions} unique actions          │          │
│  │    ⚠️ CRITICAL: Previous actions NOT advancing milestones!    │          │
│  │    Try DIFFERENT directions or explore new areas!"            │          │
│  └───────────────────────────────────────────────────────────────┘          │
│                                                                             │
│  Key Features:                                                              │
│  • Multi-source detection: Time + Steps + Location + VLM patterns           │
│  • Server data integration: Uses ground-truth location/map data             │
│  • VLM pattern analysis: Detects when guidance becomes repetitive           │
│  • Context-aware penalties: Moderate to avoid discouraging exploration       │
│  • Enhanced guidance: VLM receives comprehensive stagnation context          │
│                                                                             │
│  Architecture Impact:                                                       │
│  • Stagnation info flows: Detection → VLM prompt → Alternative actions     │
│  • Maintains milestone-driven rewards: Still rewards exploration/progress   │
│  • Comprehensive monitoring: Tracks multiple signals for robust detection  │
│                                                                             │
│  Training Impact:                                                           │
│  • Detects stuck states after critical milestones (e.g., BIRCH_LAB_VISITED) │
│  • Triggers alternative navigation strategies when standard guidance fails   │
│  • Provides debugging info: Logs location, steps, VLM patterns              │
│                                                                             │
│  Publication Significance:                                                  │
│  Demonstrates importance of multi-modal stagnation detection in long-horizon│
│  RL. Shows that combining time, steps, location, and VLM patterns provides  │
│  more robust stuck detection than any single signal alone.                  │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════

