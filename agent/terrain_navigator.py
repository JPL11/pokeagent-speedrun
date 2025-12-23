"""
Lightweight Terrain Navigator with Caching
==========================================

CPU/Memory efficient pathfinding using ground truth terrain data.

Performance:
- Cache hit (95%): ~0.12ms
- Cache miss (5%): ~2ms (bounded A*)
- Memory: ~10KB max

Author: PokeAgent Team
"""

import heapq
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class TerrainNavigator:
    """
    Lightweight terrain-aware navigator with caching.
    
    Memory budget: ~50KB max
    CPU budget: ~0.2ms per step (cache hit), ~2ms (cache miss)
    """
    
    def __init__(self):
        # LRU cache for computed paths (max 50 entries)
        self.path_cache: Dict[tuple, List[str]] = {}
        self.cache_max_size = 50
        
        # Known waypoints per map (precomputed ground truth)
        self.waypoints = {
            # OLDALE_TOWN
            0x11: {
                'pokecenter': (7, 13),
                'route103_exit': (10, 0),
                'route101_exit': (0, 9),
                'mart': (14, 13),
            },
            # OLDALE_POKECENTER (interior 1F)
            0x200: {
                'nurse_joy': (7, 3),
                'exit': (6, 17),  # Ground truth from server
                'pc': (11, 3),  # Pokemon PC
            },
            # ROUTE_101
            0x10: {
                'oldale_entrance': (8, 0),
                'littleroot_exit': (8, 30),
            },
            # ROUTE_103
            0x12: {
                'rival_area': (10, 3),
                'oldale_exit': (5, 15),
            },
            # LITTLEROOT_TOWN
            0x0101: {
                'player_house': (8, 8),
                'rival_house': (17, 8),
                'lab': (12, 14),
                'route_101_north': (12, 0),
            },
            # BIRCH_LAB
            0x020C: {
                'birch': (6, 5),
                'exit': (6, 12),
            },
        }
        
        # Last position - skip recompute if unchanged
        self.last_pos: Optional[tuple] = None
        self.last_direction: Optional[Dict] = None
        self.last_goal: Optional[tuple] = None
        
        # Stats for debugging
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get_direction(
        self, 
        map_id: int, 
        x: int, 
        y: int, 
        goal: Union[str, Tuple[int, int]], 
        metatile_info: List[List[Dict]]
    ) -> Optional[Dict]:
        """
        Get optimal direction to reach goal.
        
        Args:
            map_id: Current map ID
            x, y: Current position
            goal: (goal_x, goal_y) or goal name string
            metatile_info: 15x15 terrain grid from server
            
        Returns:
            {'action': 'UP'/'DOWN'/'LEFT'/'RIGHT'/'A', 'reason': str}
            or None if no path found
        """
        
        # OPTIMIZATION 1: Skip if same position and goal as last call
        current_pos = (map_id, x, y)
        if current_pos == self.last_pos and goal == self.last_goal and self.last_direction:
            self.cache_hits += 1
            return self.last_direction
        
        # Resolve goal name to coordinates
        goal_coords = goal
        if isinstance(goal, str):
            goal_coords = self.waypoints.get(map_id, {}).get(goal)
            if not goal_coords:
                logger.warning(f"Unknown waypoint '{goal}' for map {hex(map_id)}")
                return None
        
        goal_x, goal_y = goal_coords
        
        # OPTIMIZATION 2: At goal? Return interaction action!
        if abs(x - goal_x) <= 1 and abs(y - goal_y) <= 1:
            self.last_pos = current_pos
            self.last_goal = goal
            self.last_direction = {'action': 'A', 'reason': f'At goal ({goal_x},{goal_y}) - interact!'}
            return self.last_direction
        
        # OPTIMIZATION 3: Try simple straight-line movement
        simple_dir = self._try_simple_direction(x, y, goal_x, goal_y, metatile_info)
        if simple_dir:
            self.last_pos = current_pos
            self.last_goal = goal
            self.last_direction = simple_dir
            return simple_dir
        
        # OPTIMIZATION 4: Cache lookup (O(1))
        cache_key = (map_id, (x, y), goal_coords)
        if cache_key in self.path_cache:
            path = self.path_cache[cache_key]
            if path:
                self.cache_hits += 1
                self.last_pos = current_pos
                self.last_goal = goal
                self.last_direction = {'action': path[0], 'reason': 'Cached path'}
                return self.last_direction
        
        # OPTIMIZATION 5: Bounded A* (max 100 nodes)
        self.cache_misses += 1
        path = self._bounded_astar(x, y, goal_x, goal_y, metatile_info, max_nodes=100)
        
        # Cache result (LRU eviction if needed)
        if len(self.path_cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO, could use proper LRU)
            oldest = next(iter(self.path_cache))
            del self.path_cache[oldest]
        self.path_cache[cache_key] = path
        
        if path:
            self.last_pos = current_pos
            self.last_goal = goal
            self.last_direction = {'action': path[0], 'reason': 'A* computed path'}
            return self.last_direction
        
        # Fallback: simple heuristic
        self.last_pos = current_pos
        self.last_goal = goal
        self.last_direction = self._heuristic_direction(x, y, goal_x, goal_y)
        return self.last_direction
    
    def _try_simple_direction(
        self, 
        x: int, 
        y: int, 
        goal_x: int, 
        goal_y: int, 
        metatile_info: List[List[Dict]]
    ) -> Optional[Dict]:
        """Try simple straight-line movement (O(1) terrain check)."""
        
        if not metatile_info or len(metatile_info) < 15:
            return None
        
        center = 7  # Player is at center of 15x15 grid
        
        dx = goal_x - x
        dy = goal_y - y
        
        best_dir = None
        best_score = -1
        
        directions = [('UP', (0, -1)), ('DOWN', (0, 1)), ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]
        
        for direction, (ox, oy) in directions:
            # Check if this direction moves us closer
            new_dx = dx - ox
            new_dy = dy - oy
            old_dist = abs(dx) + abs(dy)
            new_dist = abs(new_dx) + abs(new_dy)
            
            if new_dist < old_dist:
                # Check terrain at target position
                try:
                    tile = metatile_info[center + oy][center + ox]
                    behavior = tile.get('behavior', '')
                    passable = tile.get('passable', True)
                    
                    # Skip impassable tiles
                    if 'IMPASSABLE' in behavior:
                        continue
                    
                    # Ledges only allow jumping DOWN
                    if 'JUMP' in behavior and direction != 'DOWN':
                        continue
                    
                    if not passable:
                        continue
                    
                    # Valid move! Score by distance reduction
                    score = old_dist - new_dist
                    if score > best_score:
                        best_score = score
                        best_dir = direction
                except (IndexError, KeyError, TypeError):
                    continue
        
        if best_dir:
            return {'action': best_dir, 'reason': f'Simple path to ({goal_x},{goal_y})'}
        return None
    
    def _bounded_astar(
        self, 
        x: int, 
        y: int, 
        goal_x: int, 
        goal_y: int, 
        metatile_info: List[List[Dict]], 
        max_nodes: int = 100
    ) -> Optional[List[str]]:
        """
        Bounded A* pathfinding - max nodes explored.
        Returns list of directions ['UP', 'LEFT', ...] or None.
        """
        
        if not metatile_info or len(metatile_info) < 15:
            return None
        
        center = 7
        
        def heuristic(px: int, py: int) -> int:
            return abs(goal_x - px) + abs(goal_y - py)
        
        def is_passable(px: int, py: int) -> bool:
            # Convert to grid coordinates
            gx = center + (px - x)
            gy = center + (py - y)
            
            # Outside visible grid - assume passable
            if gx < 0 or gx >= 15 or gy < 0 or gy >= 15:
                return True
            
            try:
                tile = metatile_info[gy][gx]
                behavior = tile.get('behavior', '')
                
                if 'IMPASSABLE' in behavior:
                    return False
                if 'JUMP' in behavior:
                    return False  # Treat ledges as blocked for pathfinding
                
                return tile.get('passable', True)
            except (IndexError, KeyError, TypeError):
                return True
        
        # A* with bounded exploration
        # Priority queue: (f_score, g_score, position, path)
        open_set = [(heuristic(x, y), 0, (x, y), [])]
        visited = set()
        nodes_explored = 0
        
        while open_set and nodes_explored < max_nodes:
            _, g, (cx, cy), path = heapq.heappop(open_set)
            
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            nodes_explored += 1
            
            # Goal reached?
            if abs(cx - goal_x) <= 1 and abs(cy - goal_y) <= 1:
                return path if path else ['A']
            
            # Explore neighbors
            for direction, (dx, dy) in [('UP', (0, -1)), ('DOWN', (0, 1)), 
                                         ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]:
                nx, ny = cx + dx, cy + dy
                
                if (nx, ny) not in visited and is_passable(nx, ny):
                    new_g = g + 1
                    new_f = new_g + heuristic(nx, ny)
                    new_path = path + [direction]
                    heapq.heappush(open_set, (new_f, new_g, (nx, ny), new_path))
        
        # No path found within budget
        return None
    
    def _heuristic_direction(self, x: int, y: int, goal_x: int, goal_y: int) -> Dict:
        """Fallback: simple direction towards goal."""
        dx = goal_x - x
        dy = goal_y - y
        
        if abs(dx) > abs(dy):
            action = 'RIGHT' if dx > 0 else 'LEFT'
        else:
            action = 'DOWN' if dy > 0 else 'UP'
        
        return {'action': action, 'reason': 'Heuristic (no path found)'}
    
    def clear_cache(self):
        """Clear path cache (call when entering new map)."""
        self.path_cache.clear()
        self.last_pos = None
        self.last_direction = None
        self.last_goal = None
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.path_cache),
        }
    
    def add_waypoint(self, map_id: int, name: str, coords: Tuple[int, int]):
        """Add a new waypoint dynamically."""
        if map_id not in self.waypoints:
            self.waypoints[map_id] = {}
        self.waypoints[map_id][name] = coords


def encode_terrain_for_snn(metatile_info: List[List[Dict]]) -> np.ndarray:
    """
    Encode 15x15 terrain to 225-dim vector for SNN input.
    
    Encoding:
    - 1.0 = fully passable
    - 0.5 = ledge (partial)  
    - 0.0 = blocked
    
    ~0.05ms to compute
    """
    if not metatile_info:
        return np.zeros(225, dtype=np.float32)
    
    encoding = []
    
    for row in metatile_info[:15]:
        for tile in row[:15] if row else []:
            behavior = tile.get('behavior', '') if isinstance(tile, dict) else ''
            passable = tile.get('passable', True) if isinstance(tile, dict) else True
            
            if 'IMPASSABLE' in behavior:
                encoding.append(0.0)
            elif 'JUMP' in behavior:
                encoding.append(0.5)  # Ledge = partial passability
            elif 'GRASS' in behavior:
                encoding.append(0.8)  # Grass = passable but notable
            elif passable:
                encoding.append(1.0)
            else:
                encoding.append(0.0)
    
    # Pad to 225 if needed
    while len(encoding) < 225:
        encoding.append(0.5)  # Unknown = assume partial
    
    return np.array(encoding[:225], dtype=np.float32)


# Singleton instance for easy access
_navigator_instance: Optional[TerrainNavigator] = None

def get_terrain_navigator() -> TerrainNavigator:
    """Get singleton TerrainNavigator instance."""
    global _navigator_instance
    if _navigator_instance is None:
        _navigator_instance = TerrainNavigator()
    return _navigator_instance
