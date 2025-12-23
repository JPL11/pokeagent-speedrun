"""
Novelty Detection for Pokemon Speedrun
Inspired by PokemonRedExperiments V2 coordinate-based exploration
https://github.com/PWhiddy/PokemonRedExperiments

Key improvements over basic SNN:
1. Coordinate-based novelty (fast, interpretable)
2. Frame differencing (simpler than SNN)
3. State hash tracking (detect loops)
4. Hybrid scoring (combines multiple signals)
"""

import numpy as np
import hashlib
from typing import Dict, Tuple, Optional
from collections import defaultdict
import time


class NoveltyDetector:
    """
    Fast novelty detection for Pokemon speedrun.
    Combines coordinate tracking, frame differencing, and state hashing.
    """
    
    def __init__(
        self,
        coord_weight: float = 1.0,
        frame_diff_weight: float = 0.3,
        state_hash_weight: float = 0.5,
        coord_decay: float = 0.95,  # Decay visit counts over time
        novelty_threshold: float = 0.3,  # What counts as "novel"
    ):
        # Weights for different novelty signals
        self.coord_weight = coord_weight
        self.frame_diff_weight = frame_diff_weight
        self.state_hash_weight = state_hash_weight
        self.coord_decay = coord_decay
        self.novelty_threshold = novelty_threshold
        
        # Coordinate tracking (inspired by PokemonRedExperiments V2)
        self.seen_coords: Dict[str, int] = {}  # {f"x:{x}_y:{y}_m:{map}": visit_count}
        self.unique_coord_count = 0
        self.total_steps = 0
        
        # Frame differencing
        self.last_frame: Optional[np.ndarray] = None
        self.frame_diff_history = []
        self.max_frame_diff_history = 10
        
        # State hash tracking (detect loops)
        self.seen_state_hashes: Dict[str, int] = {}  # {state_hash: visit_count}
        self.recent_state_hashes = []
        self.max_state_hash_history = 20
        
        # Metrics
        self.novelty_scores = []
        self.max_novelty_history = 100
        
        # Timing
        self.last_decay_time = time.time()
        self.decay_interval = 60.0  # Decay every 60 seconds
        
    def _make_coord_key(self, x: int, y: int, map_id: int) -> str:
        """Create coordinate key (inspired by PokemonRedExperiments)"""
        return f"x:{x}_y:{y}_m:{map_id}"
    
    def _make_state_hash(self, coords: Tuple[int, int, int], dialogue: str = "") -> str:
        """Create hash of current state"""
        state_str = f"{coords[0]},{coords[1]},{coords[2]}|{dialogue[:50]}"
        return hashlib.md5(state_str.encode()).hexdigest()[:16]
    
    def _compute_frame_diff(self, frame: np.ndarray) -> float:
        """Compute frame difference (simpler than SNN, more effective)"""
        if self.last_frame is None:
            self.last_frame = frame
            return 1.0  # First frame is novel
        
        # Resize to small size for fast comparison
        small_frame = self._resize_frame(frame)
        small_last = self._resize_frame(self.last_frame)
        
        # Compute absolute difference
        diff = np.abs(small_frame.astype(float) - small_last.astype(float))
        diff_score = np.mean(diff) / 255.0  # Normalize to 0-1
        
        # Update
        self.last_frame = frame
        self.frame_diff_history.append(diff_score)
        if len(self.frame_diff_history) > self.max_frame_diff_history:
            self.frame_diff_history.pop(0)
        
        return diff_score
    
    def _resize_frame(self, frame: np.ndarray, size: Tuple[int, int] = (40, 40)) -> np.ndarray:
        """Resize frame for fast comparison"""
        if len(frame.shape) == 3:
            # Convert to grayscale if color
            frame = np.mean(frame, axis=2).astype(np.uint8)
        
        # Simple downsampling (faster than cv2.resize)
        h, w = frame.shape
        target_h, target_w = size
        stride_h, stride_w = h // target_h, w // target_w
        
        return frame[::stride_h, ::stride_w][:target_h, :target_w]
    
    def update(
        self,
        coords: Tuple[int, int, int],  # (x, y, map_id)
        frame: Optional[np.ndarray] = None,
        dialogue: str = "",
    ) -> Dict[str, float]:
        """
        Update novelty detector with new observation.
        
        Returns:
            dict with novelty scores and metrics
        """
        self.total_steps += 1
        
        # 1. Coordinate novelty (inspired by PokemonRedExperiments V2)
        coord_key = self._make_coord_key(coords[0], coords[1], coords[2])
        
        if coord_key not in self.seen_coords:
            self.unique_coord_count += 1
            self.seen_coords[coord_key] = 1
            coord_novelty = 1.0  # Completely new coordinate!
        else:
            self.seen_coords[coord_key] += 1
            visit_count = self.seen_coords[coord_key]
            # Diminishing returns for revisiting same spot
            coord_novelty = 1.0 / (1.0 + np.log(visit_count))
        
        # 2. Frame difference novelty
        frame_novelty = 0.5  # Default if no frame provided
        if frame is not None:
            frame_novelty = self._compute_frame_diff(frame)
        
        # 3. State hash novelty (detect loops)
        state_hash = self._make_state_hash(coords, dialogue)
        
        if state_hash not in self.seen_state_hashes:
            self.seen_state_hashes[state_hash] = 1
            state_novelty = 1.0
        else:
            self.seen_state_hashes[state_hash] += 1
            visit_count = self.seen_state_hashes[state_hash]
            # Strong penalty for repeating exact same state
            state_novelty = 1.0 / (1.0 + visit_count * 2.0)
        
        # Track recent state hashes for loop detection
        self.recent_state_hashes.append(state_hash)
        if len(self.recent_state_hashes) > self.max_state_hash_history:
            self.recent_state_hashes.pop(0)
        
        # 4. Combine novelty signals
        total_novelty = (
            self.coord_weight * coord_novelty +
            self.frame_diff_weight * frame_novelty +
            self.state_hash_weight * state_novelty
        )
        
        # Normalize by total weight
        total_weight = self.coord_weight + self.frame_diff_weight + self.state_hash_weight
        normalized_novelty = total_novelty / total_weight
        
        # Track novelty scores
        self.novelty_scores.append(normalized_novelty)
        if len(self.novelty_scores) > self.max_novelty_history:
            self.novelty_scores.pop(0)
        
        # Periodic decay of visit counts (encourage revisiting old areas)
        if time.time() - self.last_decay_time > self.decay_interval:
            self._decay_visit_counts()
            self.last_decay_time = time.time()
        
        return {
            "total_novelty": normalized_novelty,
            "coord_novelty": coord_novelty,
            "frame_novelty": frame_novelty,
            "state_novelty": state_novelty,
            "is_novel": normalized_novelty > self.novelty_threshold,
            "unique_coords": self.unique_coord_count,
            "exploration_progress": self.unique_coord_count * 0.1,  # PokemonRedExperiments style
        }
    
    def _decay_visit_counts(self):
        """Decay visit counts over time (encourage revisiting)"""
        for key in self.seen_coords:
            self.seen_coords[key] = max(1, int(self.seen_coords[key] * self.coord_decay))
        
        for key in self.seen_state_hashes:
            self.seen_state_hashes[key] = max(1, int(self.seen_state_hashes[key] * self.coord_decay))
    
    def detect_loop(self, window: int = 10) -> bool:
        """Detect if agent is stuck in a loop"""
        if len(self.recent_state_hashes) < window:
            return False
        
        recent = self.recent_state_hashes[-window:]
        unique_states = len(set(recent))
        
        # If less than 40% unique states in recent history = loop
        return unique_states < window * 0.4
    
    def get_summary(self) -> str:
        """Get human-readable summary for VLM prompt"""
        avg_novelty = np.mean(self.novelty_scores) if self.novelty_scores else 0.5
        recent_novelty = self.novelty_scores[-1] if self.novelty_scores else 0.5
        
        is_in_loop = self.detect_loop()
        
        exploration_rate = self.unique_coord_count / max(1, self.total_steps)
        
        summary = f"""🔍 NOVELTY ANALYSIS:
• Exploration Progress: {self.unique_coord_count} unique locations visited
• Exploration Rate: {exploration_rate:.2%} (unique coords / total steps)
• Current Novelty: {recent_novelty:.2f} / 1.0 {'✨ NOVEL' if recent_novelty > self.novelty_threshold else '⚠️ FAMILIAR'}
• Average Novelty: {avg_novelty:.2f}
• Loop Detected: {'🔄 YES - Try something different!' if is_in_loop else '✅ NO'}
• Total Steps: {self.total_steps}

💡 EXPLORATION HINT: {'Explore new areas for higher rewards!' if recent_novelty < 0.4 else 'Good progress, keep exploring!'}"""
        
        return summary
    
    def reset(self):
        """Reset novelty detector (for new episode)"""
        self.seen_coords.clear()
        self.seen_state_hashes.clear()
        self.recent_state_hashes.clear()
        self.novelty_scores.clear()
        self.frame_diff_history.clear()
        self.last_frame = None
        self.unique_coord_count = 0
        self.total_steps = 0


def create_novelty_detector(
    coord_weight: float = 1.0,
    frame_diff_weight: float = 0.3,
    state_hash_weight: float = 0.5,
) -> NoveltyDetector:
    """Factory function to create novelty detector"""
    return NoveltyDetector(
        coord_weight=coord_weight,
        frame_diff_weight=frame_diff_weight,
        state_hash_weight=state_hash_weight,
    )

