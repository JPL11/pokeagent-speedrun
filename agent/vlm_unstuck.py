"""
VLM-based Unstuck Mechanism for RL Agent
=========================================

This module provides VLM reasoning to detect and resolve stuck states
without relying on hard-coded heuristics. Inspired by PokemonRedExperiments
but with enhanced VLM-driven decision making.

Key Features:
1. VLM analyzes game state to detect stuck patterns
2. Provides context-aware suggestions to RL agent (not hard overrides)
3. Learns from stuck patterns to improve over time
4. Rewards RL agent for following good VLM advice

Philosophy:
- VLM provides REASONING and CONTEXT
- RL agent learns to USE this context to make better actions
- Minimizes hard-coded overrides, maximizes learned behavior
"""

import logging
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class StuckPattern:
    """Represents a detected stuck pattern"""
    pattern_type: str  # 'coordinate_loop', 'dialogue_spam', 'animation_farm', 'menu_trap'
    confidence: float
    duration: int  # How many steps this pattern has persisted
    context: str  # VLM's reasoning about why this is stuck
    suggested_action: Optional[str] = None
    reward_modifier: float = 0.0  # Bonus/penalty to guide RL learning


class VLMUnstuckDetector:
    """
    Uses VLM reasoning to detect stuck states and guide RL exploration.
    
    Unlike hard-coded heuristics, this uses VLM to:
    1. Understand CONTEXT (what is the agent trying to do?)
    2. Detect STUCK patterns (is progress being made?)
    3. Suggest EXPLORATION strategies (what to try next?)
    4. Provide REWARD guidance (did the suggestion help?)
    """
    
    def __init__(
        self,
        vlm,
        check_frequency: int = 30,  # Check every 30 steps
        coordinate_history_size: int = 50,
        enable_suggestions: bool = True,
        verbose: bool = True
    ):
        """
        Args:
            vlm: VLM instance for reasoning
            check_frequency: How often to ask VLM for stuck analysis
            coordinate_history_size: How many coords to track
            enable_suggestions: Whether to provide action suggestions
            verbose: Log detailed reasoning
        """
        self.vlm = vlm
        self.check_frequency = check_frequency
        self.enable_suggestions = enable_suggestions
        self.verbose = verbose
        
        # State tracking (like PokemonRedExperiments)
        self.coordinate_history = deque(maxlen=coordinate_history_size)
        self.action_history = deque(maxlen=50)
        self.dialogue_history = deque(maxlen=20)
        self.stuck_patterns: List[StuckPattern] = []
        
        # Metrics
        self.steps_since_check = 0
        self.total_stuck_detections = 0
        self.successful_unstucks = 0
        
        # Learning: Track which VLM suggestions helped
        self.suggestion_outcomes = {}  # suggestion -> (success_count, total_count)
        
    def update_state(self, game_state: Dict[str, Any], action: str):
        """Update state tracking with latest game data"""
        coords = game_state.get('coordinates', (0, 0))
        map_id = game_state.get('map_id', 0)
        coord_tuple = (coords[0], coords[1], map_id)
        
        self.coordinate_history.append(coord_tuple)
        self.action_history.append(action)
        
        dialogue = game_state.get('dialogue', '')
        if dialogue and len(dialogue) > 3:
            self.dialogue_history.append(dialogue)
        
        self.steps_since_check += 1
    
    def check_stuck(self, frame, game_state: Dict[str, Any]) -> Optional[StuckPattern]:
        """
        Ask VLM to analyze if agent is stuck and why.
        
        This replaces hard-coded heuristics with VLM reasoning.
        
        Returns:
            StuckPattern if stuck is detected, None otherwise
        """
        if self.steps_since_check < self.check_frequency:
            return None
        
        self.steps_since_check = 0
        
        # Gather context for VLM
        recent_coords = list(self.coordinate_history)[-20:]
        unique_coords = len(set(recent_coords))
        recent_actions = list(self.action_history)[-20:]
        recent_dialogues = list(self.dialogue_history)[-5:]
        
        # Build VLM prompt with context
        prompt = f"""You are analyzing a Pokemon Emerald RL agent's behavior to detect stuck patterns.

**Current State:**
- Location: {game_state.get('location', 'UNKNOWN')}
- Coordinates: {game_state.get('coordinates', (0, 0))}
- Map ID: {game_state.get('map_id', 0)}

**Recent Behavior (last 20 steps):**
- Unique coordinates visited: {unique_coords} out of 20
- Recent actions: {', '.join(recent_actions[-10:])}
- Recent dialogues: {len(recent_dialogues)} unique

**Analysis Questions:**
1. Is the agent stuck? (YES/NO)
2. If YES, what type of stuck pattern? (coordinate_loop, dialogue_spam, menu_trap, other)
3. Why is the agent stuck? (brief explanation)
4. Confidence that agent is stuck? (0.0-1.0)

**Response Format (JSON-like):**
{{
    "is_stuck": true/false,
    "pattern_type": "...",
    "reasoning": "...",
    "confidence": 0.0-1.0,
    "suggested_exploration": "..." (optional: what should agent try?)
}}

Respond with analysis:"""

        try:
            response = self.vlm.get_query(frame, prompt, "unstuck_detection")
            
            # Parse VLM response
            stuck_pattern = self._parse_vlm_response(response, recent_coords, recent_actions)
            
            if stuck_pattern and stuck_pattern.confidence > 0.6:
                self.total_stuck_detections += 1
                self.stuck_patterns.append(stuck_pattern)
                
                if self.verbose:
                    logger.info(f"🚨 VLM detected stuck: {stuck_pattern.pattern_type}")
                    logger.info(f"   Confidence: {stuck_pattern.confidence:.2f}")
                    logger.info(f"   Reasoning: {stuck_pattern.context}")
                    if stuck_pattern.suggested_action:
                        logger.info(f"   Suggestion: {stuck_pattern.suggested_action}")
                
                return stuck_pattern
            
            return None
            
        except Exception as e:
            logger.debug(f"VLM unstuck check failed: {e}")
            return None
    
    def _parse_vlm_response(self, response: str, coords: List, actions: List) -> Optional[StuckPattern]:
        """Parse VLM response into StuckPattern"""
        try:
            # Extract key information from response
            is_stuck = 'true' in response.lower() or 'yes' in response.lower()
            
            if not is_stuck:
                return None
            
            # Extract pattern type
            pattern_type = "unknown"
            if 'coordinate_loop' in response.lower() or 'same area' in response.lower():
                pattern_type = "coordinate_loop"
            elif 'dialogue' in response.lower() and 'spam' in response.lower():
                pattern_type = "dialogue_spam"
            elif 'menu' in response.lower():
                pattern_type = "menu_trap"
            elif 'animation' in response.lower():
                pattern_type = "animation_farm"
            
            # Extract confidence (look for number between 0-1)
            import re
            confidence_match = re.search(r'0\.\d+|1\.0', response)
            confidence = float(confidence_match.group()) if confidence_match else 0.7
            
            # Extract reasoning (first few sentences)
            reasoning_lines = response.split('\n')
            reasoning = ' '.join(reasoning_lines[:3])[:200]
            
            # Extract suggestion if present
            suggested_action = None
            if 'try' in response.lower():
                # Look for action suggestions
                for action in ['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'START']:
                    if action.lower() in response.lower():
                        suggested_action = action
                        break
            
            return StuckPattern(
                pattern_type=pattern_type,
                confidence=confidence,
                duration=len(coords),
                context=reasoning,
                suggested_action=suggested_action,
                reward_modifier=-0.5  # Small penalty for being stuck
            )
            
        except Exception as e:
            logger.debug(f"Failed to parse VLM response: {e}")
            return None
    
    def get_suggestion_reward_bonus(self, action: str, was_unstuck: bool) -> float:
        """
        Calculate reward bonus for following VLM suggestion.
        
        This teaches the RL agent to trust VLM guidance when stuck.
        """
        if not self.stuck_patterns:
            return 0.0
        
        latest_pattern = self.stuck_patterns[-1]
        
        # If agent followed VLM suggestion
        if latest_pattern.suggested_action and action == latest_pattern.suggested_action:
            # Track outcome
            suggestion_key = f"{latest_pattern.pattern_type}_{action}"
            if suggestion_key not in self.suggestion_outcomes:
                self.suggestion_outcomes[suggestion_key] = [0, 0]
            
            if was_unstuck:
                # Success! Give reward bonus
                self.suggestion_outcomes[suggestion_key][0] += 1
                self.successful_unstucks += 1
                bonus = 2.0
                
                if self.verbose:
                    logger.info(f"✅ VLM suggestion worked! Bonus: +{bonus}")
                
                return bonus
            else:
                # Suggestion didn't help yet
                self.suggestion_outcomes[suggestion_key][1] += 1
                return 0.0
        
        return 0.0
    
    def was_unstuck(self) -> bool:
        """Check if agent successfully got unstuck"""
        if len(self.coordinate_history) < 10:
            return False
        
        # Check if exploration improved
        recent_10 = list(self.coordinate_history)[-10:]
        unique_recent = len(set(recent_10))
        
        # If seeing many new coordinates, we're unstuck!
        return unique_recent >= 7
    
    def get_stats(self) -> Dict[str, Any]:
        """Get unstuck detection statistics"""
        success_rate = 0.0
        if self.total_stuck_detections > 0:
            success_rate = self.successful_unstucks / self.total_stuck_detections
        
        return {
            'total_stuck_detections': self.total_stuck_detections,
            'successful_unstucks': self.successful_unstucks,
            'success_rate': success_rate,
            'suggestion_outcomes': self.suggestion_outcomes
        }


class VLMGuidedRewardShaper:
    """
    Uses VLM reasoning to shape rewards dynamically based on game context.
    
    Instead of fixed reward weights, VLM helps determine:
    - What is important RIGHT NOW?
    - What progress has been made?
    - What should be rewarded more/less?
    """
    
    def __init__(self, vlm, check_frequency: int = 100):
        self.vlm = vlm
        self.check_frequency = check_frequency
        self.steps_since_check = 0
        self.current_priorities = {
            'exploration': 1.0,
            'dialogue': 1.0,
            'events': 1.0
        }
    
    def update_priorities(self, frame, game_state: Dict[str, Any]):
        """Ask VLM to update reward priorities based on game progress"""
        self.steps_since_check += 1
        
        if self.steps_since_check < self.check_frequency:
            return
        
        self.steps_since_check = 0
        
        prompt = f"""You are helping tune rewards for a Pokemon Emerald RL agent.

**Current State:**
- Location: {game_state.get('location', 'UNKNOWN')}
- Recent events: {len(game_state.get('events', []))}

**Question:** What should the agent prioritize RIGHT NOW?
Rate importance (0.0-2.0) for:
1. Exploration (finding new areas)
2. Dialogue (talking to NPCs)
3. Events (achieving milestones)

**Format:**
exploration: X.X
dialogue: X.X
events: X.X
reasoning: brief explanation

Respond:"""
        
        try:
            response = self.vlm.get_text_query(prompt, "reward_priorities")
            
            # Parse priorities from response
            import re
            for priority in ['exploration', 'dialogue', 'events']:
                match = re.search(f'{priority}:\\s*(\\d+\\.\\d+)', response, re.IGNORECASE)
                if match:
                    self.current_priorities[priority] = float(match.group(1))
            
            logger.info(f"🎯 Updated reward priorities: {self.current_priorities}")
            
        except Exception as e:
            logger.debug(f"Failed to update reward priorities: {e}")
    
    def get_shaped_reward(self, base_reward: float, reward_type: str) -> float:
        """Apply VLM-guided reward shaping"""
        priority = self.current_priorities.get(reward_type, 1.0)
        return base_reward * priority
