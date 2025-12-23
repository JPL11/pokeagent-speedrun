"""
k-NN Episodic Memory Module

Purpose: Retrieve similar past states to augment SNN temporal reasoning
- NOT a replacement for SNN (SNN = temporal dependencies, k-NN = similarity search)
- Acts as "recall what worked before in similar situations"

Integration:
    episodic_recall = knn.query(current_state)
    augmented_context = concat(snn_embed, episodic_recall)
    policy_input = augmented_context
"""

import numpy as np
from collections import deque
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    k-NN based episodic memory for navigation, battles, and quest patterns.
    
    **Context-Aware Design**: Separate memories for different domains
        - Navigation: (cnn_feat, story_vec, waypoint_hint) → action, reward, milestone
        - Battle: (battle_vec, hp_state, type_advantages) → action, outcome
        - Quest: (dialogue_state, story_context) → next_milestone
    
    **Confidence Gating**: Returns (knn_vector, confidence) tuple
        - confidence = mean cosine similarity of k nearest neighbors
        - Allows SNN/policy to down-weight uncertain recalls
    
    Returns similar past experiences to augment SNN temporal reasoning.
    """
    
    def __init__(self, max_size: int = 10000, k: int = 5):
        """
        Args:
            max_size: Maximum number of experiences to store per domain
            k: Number of nearest neighbors to retrieve
        """
        self.max_size = max_size
        self.k = k
        
        # Separate memory buffers for different domains (context-aware)
        self.nav_memory = deque(maxlen=max_size)      # Navigation experiences
        self.battle_memory = deque(maxlen=max_size)    # Battle experiences
        self.quest_memory = deque(maxlen=max_size)     # Quest/story patterns
        
        logger.info(f"🧠 EpisodicMemory initialized: max_size={max_size}, k={k} (context-aware)")
    
    def store_navigation_experience(
        self,
        cnn_features: np.ndarray,       # Local visual features
        story_vec: np.ndarray,          # Current quest context
        waypoint_hint: np.ndarray,      # SLAM direction hint
        action_taken: int,              # Action that was taken
        reward_received: float,         # Immediate reward
        milestone_reached: Optional[str] = None  # Milestone completed (if any)
    ):
        """Store a navigation experience for later recall."""
        # Create compact key
        nav_key = np.concatenate([
            cnn_features.flatten()[:32],    # Downsample CNN to 32-dim
            story_vec[:16],                 # Quest context 16-dim
            waypoint_hint[:8]               # SLAM hint 8-dim
        ])  # Total: 56-dim key
        
        experience = {
            'key': nav_key,
            'action': action_taken,
            'reward': reward_received,
            'milestone': milestone_reached,
            'type': 'navigation'
        }
        
        self.nav_memory.append(experience)
    
    def store_battle_experience(
        self,
        battle_vec: np.ndarray,         # Battle strategy features
        hp_state: np.ndarray,           # [my_hp_ratio, opp_hp_ratio, my_count, opp_count]
        type_advantages: np.ndarray,    # Type matchup scores
        action_taken: int,              # Move/switch action
        outcome: str,                   # 'won', 'lost', 'fainted', 'switched'
        gym_battle: bool = False
    ):
        """Store a battle experience for later recall."""
        battle_key = np.concatenate([
            battle_vec[:16],                # Battle strategy 16-dim
            hp_state[:4],                   # HP ratios 4-dim
            type_advantages[:8]             # Type matchups 8-dim
        ])  # Total: 28-dim key
        
        experience = {
            'key': battle_key,
            'action': action_taken,
            'outcome': outcome,
            'gym_battle': gym_battle,
            'type': 'battle'
        }
        
        self.battle_memory.append(experience)
    
    def store_quest_experience(
        self,
        dialogue_state: np.ndarray,     # Current dialogue/quest encoding
        story_context: np.ndarray,      # Milestone flags
        next_milestone: Optional[str] = None
    ):
        """Store a quest transition pattern."""
        quest_key = np.concatenate([
            dialogue_state[:16],
            story_context[:16]
        ])  # Total: 32-dim key
        
        experience = {
            'key': quest_key,
            'next_milestone': next_milestone,
            'type': 'quest'
        }
        
        self.quest_memory.append(experience)
    
    def query_navigation(
        self,
        current_cnn_features: np.ndarray,
        current_story_vec: np.ndarray,
        current_waypoint_hint: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Query k-NN for similar past navigation states.
        
        Returns:
            {
                'suggested_actions': array of k most common actions in similar states,
                'action_weights': probability distribution over actions,
                'avg_reward': average reward from similar past states,
                'milestone_likelihood': probability a milestone was reached
            }
        """
        if len(self.nav_memory) < self.k:
            # Not enough memories yet
            return {
                'suggested_actions': np.zeros(self.k, dtype=np.int32),
                'action_weights': np.ones(8) / 8,  # Uniform over 8 actions
                'avg_reward': 0.0,
                'milestone_likelihood': 0.0
            }
        
        # Create query key
        query_key = np.concatenate([
            current_cnn_features.flatten()[:32],
            current_story_vec[:16],
            current_waypoint_hint[:8]
        ])
        
        # Find k nearest neighbors (L2 distance)
        distances = []
        for exp in self.nav_memory:
            dist = np.linalg.norm(query_key - exp['key'])
            distances.append((dist, exp))
        
        # Get k closest
        distances.sort(key=lambda x: x[0])
        neighbors = [exp for (_, exp) in distances[:self.k]]
        
        # Aggregate information from neighbors
        actions = [exp['action'] for exp in neighbors]
        rewards = [exp['reward'] for exp in neighbors]
        milestones = [exp['milestone'] is not None for exp in neighbors]
        
        # Compute action distribution (weighted by inverse distance)
        action_counts = np.zeros(8)  # 8 possible actions
        for i, exp in enumerate(neighbors):
            weight = 1.0 / (distances[i][0] + 1e-6)  # Inverse distance weight
            action_counts[exp['action']] += weight
        action_weights = action_counts / (action_counts.sum() + 1e-6)
        
        return {
            'suggested_actions': np.array(actions, dtype=np.int32),
            'action_weights': action_weights,
            'avg_reward': np.mean(rewards) if rewards else 0.0,
            'milestone_likelihood': np.mean(milestones) if milestones else 0.0
        }
    
    def query_battle(
        self,
        current_battle_vec: np.ndarray,
        current_hp_state: np.ndarray,
        current_type_advantages: np.ndarray,
        gym_context: bool = False
    ) -> Dict[str, any]:
        """
        Query k-NN for similar past battle situations.
        
        Returns what worked in similar matchups.
        """
        if len(self.battle_memory) < self.k:
            return {
                'suggested_move': None,
                'win_rate': 0.5,
                'risk_score': 0.5
            }
        
        query_key = np.concatenate([
            current_battle_vec[:16],
            current_hp_state[:4],
            current_type_advantages[:8]
        ])
        
        # Find k nearest neighbors
        distances = []
        for exp in self.battle_memory:
            # Filter by gym context if needed
            if gym_context and not exp.get('gym_battle', False):
                continue
            
            dist = np.linalg.norm(query_key - exp['key'])
            distances.append((dist, exp))
        
        if not distances:
            return {
                'suggested_move': None,
                'win_rate': 0.5,
                'risk_score': 0.5
            }
        
        distances.sort(key=lambda x: x[0])
        neighbors = [exp for (_, exp) in distances[:self.k]]
        
        # Aggregate battle outcomes
        outcomes = [exp['outcome'] for exp in neighbors]
        win_rate = outcomes.count('won') / len(outcomes)
        loss_rate = outcomes.count('lost') / len(outcomes)
        faint_rate = outcomes.count('fainted') / len(outcomes)
        
        # Most common successful action
        winning_actions = [exp['action'] for exp in neighbors if exp['outcome'] == 'won']
        suggested_move = max(set(winning_actions), key=winning_actions.count) if winning_actions else None
        
        return {
            'suggested_move': suggested_move,
            'win_rate': win_rate,
            'risk_score': loss_rate + faint_rate  # High risk if losses/faints were common
        }
    
    def get_stuck_solution(
        self,
        current_position: Tuple[int, int],
        current_map: int,
        last_n_positions: List[Tuple[int, int]]
    ) -> Optional[str]:
        """
        Check if agent is stuck in a pattern and retrieve solution from memory.
        
        Returns:
            Hint like "try_left_route" or "go_around_south" if similar stuck pattern seen before
        """
        # Check if current position is being revisited too often
        position_counts = {}
        for pos in last_n_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        if position_counts.get(current_position, 0) > 5:
            # We're stuck! Check memory for similar situations
            for exp in reversed(self.nav_memory):
                # Look for experiences on same map with high reward (successful escape)
                if exp.get('milestone') is not None and exp['reward'] > 1.0:
                    # Found a successful navigation - suggest similar action
                    return f"similar_success_action_{exp['action']}"
        
        return None
    
    def get_episodic_augmentation_with_confidence(
        self,
        domain: str,
        query_features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """
        **Confidence-aware k-NN query**: Returns (augmentation, confidence)
        
        Args:
            domain: 'navigation', 'battle', or 'quest'
            query_features: Dict of feature arrays for k-NN query
        
        Returns:
            (32-dim augmentation vector, confidence scalar in [0,1])
            - confidence = mean inverse distance of k nearest neighbors
            - High confidence = query matches known states well
            - Low confidence = novel/unseen situation
        """
        augmentation, confidence = self._query_with_confidence(domain, query_features)
        return augmentation, confidence
    
    def _query_with_confidence(
        self,
        domain: str,
        query_features: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, float]:
        """Internal: Compute augmentation + confidence."""
        if domain == 'navigation':
            # Build query key
            query_key = np.concatenate([
                query_features['cnn_features'].flatten()[:32],
                query_features['story_vec'][:16],
                query_features['waypoint_hint'][:8]
            ])
            
            if len(self.nav_memory) < self.k:
                return np.zeros(32, dtype=np.float32), 0.0
            
            # Find k nearest neighbors with distances
            distances = []
            for exp in self.nav_memory:
                dist = np.linalg.norm(query_key - exp['key'])
                distances.append((dist, exp))
            
            distances.sort(key=lambda x: x[0])
            neighbors = [exp for (dist, exp) in distances[:self.k]]
            neighbor_dists = [dist for (dist, exp) in distances[:self.k]]
            
            # Compute confidence: mean inverse distance (normalized)
            # High confidence = small distances = familiar states
            confidence = np.mean([1.0 / (d + 0.1) for d in neighbor_dists])
            confidence = min(1.0, confidence / 10.0)  # Normalize to [0, 1]
            
            # Build augmentation from neighbors
            actions = [exp['action'] for exp in neighbors]
            rewards = [exp['reward'] for exp in neighbors]
            milestones = [exp['milestone'] is not None for exp in neighbors]
            
            action_counts = np.zeros(8)
            for i, exp in enumerate(neighbors):
                weight = 1.0 / (neighbor_dists[i] + 1e-6)
                action_counts[exp['action']] += weight
            action_weights = action_counts / (action_counts.sum() + 1e-6)
            
            augmentation = np.concatenate([
                action_weights,                     # 8-dim
                [np.mean(rewards)],                 # 1-dim
                [np.mean(milestones)],              # 1-dim
                [confidence],                       # 1-dim ★ NEW
                np.zeros(21)                        # 21-dim padding
            ]).astype(np.float32)
            
            return augmentation, confidence
        
        elif domain == 'battle':
            query_key = np.concatenate([
                query_features['battle_vec'][:16],
                query_features['hp_state'][:4],
                query_features['type_advantages'][:8]
            ])
            
            if len(self.battle_memory) < self.k:
                return np.zeros(32, dtype=np.float32), 0.0
            
            distances = []
            for exp in self.battle_memory:
                dist = np.linalg.norm(query_key - exp['key'])
                distances.append((dist, exp))
            
            distances.sort(key=lambda x: x[0])
            neighbors = [exp for (dist, exp) in distances[:self.k]]
            neighbor_dists = [dist for (dist, exp) in distances[:self.k]]
            
            confidence = np.mean([1.0 / (d + 0.1) for d in neighbor_dists])
            confidence = min(1.0, confidence / 10.0)
            
            outcomes = [exp['outcome'] for exp in neighbors]
            win_rate = outcomes.count('won') / len(outcomes)
            risk_score = (outcomes.count('lost') + outcomes.count('fainted')) / len(outcomes)
            
            augmentation = np.concatenate([
                [win_rate],
                [risk_score],
                [confidence],
                np.zeros(29)
            ]).astype(np.float32)
            
            return augmentation, confidence
        
        else:  # quest
            return np.zeros(32, dtype=np.float32), 0.0
    
    def get_episodic_augmentation(
        self,
        domain: str,
        query_features: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Main interface: Get episodic memory augmentation vector for SNN.
        **Legacy API** (for backward compatibility) - returns only augmentation, no confidence.
        
        For confidence-aware queries, use get_episodic_augmentation_with_confidence()
        
        Args:
            domain: 'navigation', 'battle', or 'quest'
            query_features: Dict of feature arrays for k-NN query
        
        Returns:
            32-dim augmentation vector to concat with SNN embedding
        """
        # Use confidence-aware version, but discard confidence for backward compatibility
        augmentation, _ = self._query_with_confidence(domain, query_features)
        return augmentation
    
    def get_stats(self) -> Dict[str, int]:
        """Return memory statistics."""
        return {
            'nav_memories': len(self.nav_memory),
            'battle_memories': len(self.battle_memory),
            'quest_memories': len(self.quest_memory),
            'total': len(self.nav_memory) + len(self.battle_memory) + len(self.quest_memory)
        }
