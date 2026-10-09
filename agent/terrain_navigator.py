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
                'pokecenter': (6, 17),  # SERVER GROUND TRUTH - warp tile at (6,17)
                'route103_exit': (10, 0),  # North exit to Route 103
                'route101_exit': (0, 9),  # South exit to Route 101
                'route102_exit': (0, 7),  # West exit to Route 102 (corrected from 0,14)
                'mart': (14, 13),
            },

            # OLDALE_POKECENTER (interior 1F)
            0x200: {
                'nurse_joy': (7, 3),
                'exit': (6, 17),  # Ground truth from server
                'pc': (11, 3),  # Pokemon PC
            },
            # OLDALE_POKECENTER 2F
            0x201: {
                'exit': (1, 6),  # Escalator exit to 1F
                'escalator': (1, 6),  # Same position
            },
            # OLDALE_MART (interior) - map_id may be 0x202 or 0x0 depending on server
            0x202: {
                'exit': (2, 8),  # Exit door (south side, door mat)
                'clerk': (2, 2),  # Shop clerk at counter
            },
            # Fallback for mart when map_id is 0 (server sometimes returns 0)
            0x0: {
                'exit': (2, 8),  # Exit door
            },

            # ROUTE_101
            0x10: {
                'oldale_entrance': (8, 0),  # North exit to Oldale
                'littleroot_exit': (8, 30),  # South exit
                'littleroot_entrance': (8, 30),  # South to Littleroot (for lab)
            },

            # ROUTE_103
            # Has ledges that create a snake path to rival
            # Rival is at approximately (10, 3) in upper area
            0x12: {
                'rival_area': (10, 3),
                'oldale_exit': (5, 15),
                # Intermediate waypoints to navigate around ledges
                'ledge_bypass_1': (8, 12),  # Go RIGHT past first ledge area
                'ledge_bypass_2': (12, 8),  # Continue RIGHT to bypass ledge zone
                'upper_path': (10, 5),      # Upper area before rival
            },
            # LITTLEROOT_TOWN (outdoor map 0x09)
            0x09: {
                'birch_lab': (6, 14),  # Birch's Lab entrance
                'player_house': (3, 8),
                'rival_house': (11, 8),
                'route101_exit': (6, 0),  # Exit north to Route 101
            },
            # LITTLEROOT_TOWN (alternate ID)
            0x0101: {
                'player_house': (8, 8),
                'rival_house': (17, 8),
                'lab': (12, 14),
                'route_101_north': (12, 0),
            },
            # ROUTE_102 (between Oldale and Petalburg)
            0x0F: {
                'oldale_exit': (28, 8),  # East exit back to Oldale
                'petalburg_entrance': (0, 8),  # West exit to Petalburg City
            },
            # BIRCH_LAB
            0x020C: {
                'birch': (6, 5),
                'exit': (6, 12),
            },
            # BIRCH_LAB (alternate ID)
            0x104: {
                'birch': (6, 5),
                'exit': (6, 12),
            },
            # === GENERIC HOUSE INTERIORS ===
            # Houses typically have exit at bottom center
            # These cover Oldale Town houses, player's house, rival's house, etc.
            0x100: {'exit': (8, 8)},  # Brendan's House 1F (Player's house - door at 8,8)
            0x101: {'exit': (8, 2)},  # Brendan's House 2F (stairs at 8,2)
            0x102: {'exit': (4, 8)},  # May's House 1F (Rival's house)  
            0x103: {'exit': (4, 8)},  # May's House 2F
            0x205: {'exit': (4, 8)},  # Oldale Town House 1
            0x206: {'exit': (4, 8)},  # Oldale Town House 2
        }
        
        # Generic house exit - used when map_id not in waypoints
        self.generic_house_exit = (4, 8)  # Most houses have door at bottom center

        # ═══════════════════════════════════════════════════════════════════════════
        # ROUTE 101 WAYPOINT SYSTEM
        # Complex terrain with ledges and trees requires incremental navigation
        # A* can't find path across entire map, so use waypoints as intermediate goals
        # ═══════════════════════════════════════════════════════════════════════════
        
        # Waypoints going NORTH (Littleroot -> Oldale)
        # Route 101 layout: ledge in middle blocks direct UP, must go around RIGHT side
        self.route101_waypoints_north = [
            # (x, y) - ordered from south to north
            (8, 28),   # South entrance from Littleroot
            (8, 20),   # Before ledge zone - can go straight up
            (10, 14),  # Ledge zone - go RIGHT first 
            (12, 10),  # Past ledge on right side - gap is here
            (10, 6),   # Upper zone - can go up again
            (8, 3),    # Near Oldale entrance
            (8, 0),    # Oldale exit (final goal)
        ]
        
        # Waypoints going SOUTH (Oldale -> Littleroot)  
        self.route101_waypoints_south = list(reversed(self.route101_waypoints_north))
        
        # ═══════════════════════════════════════════════════════════════════════════
        # ROUTE 103 - DISABLED WAYPOINT SYSTEM
        # Let A* navigate directly to rival using visual map passability data
        # Waypoints were causing agent to go wrong direction
        # ═══════════════════════════════════════════════════════════════════════════
        
        # DISABLED: Empty lists = no waypoint-based navigation, use direct A* instead
        self.route103_waypoints_north = []
        self.route103_east_recovery = []
        self.route103_waypoints_south = []
        
        # ═══════════════════════════════════════════════════════════════════════════
        # OLDALE TOWN WAYPOINT SYSTEM
        # There's a building at (3-6, y=14) that blocks direct path to PokeCenter (6,17)
        # Must go around: LEFT past building, DOWN to y=17, then RIGHT to x=6
        # ═══════════════════════════════════════════════════════════════════════════
        self.oldale_pokecenter_waypoints = [
            (2, 13),   # First: go LEFT past the building at x=3-6
            (2, 17),   # Then go DOWN to same y as PokeCenter
            (6, 17),   # Finally go RIGHT to PokeCenter entrance
        ]
        
        # ═══════════════════════════════════════════════════════════════════════════
        # PLAYER'S HOUSE 1F WAYPOINT SYSTEM (May's 0x102 or Brendan's 0x100)
        # Moving Pokemon/NPC blocks direct path to stairs at (2,2)
        # Must go: DOWN to bottom walkway, LEFT along bottom, then UP to stairs
        # Layout: Stairs at (2,2), NPC moves in middle area
        # ═══════════════════════════════════════════════════════════════════════════
        self.house_1f_stairs_waypoints = [
            (10, 7),   # First: Stay on bottom walkway (if coming from spawn at 10,8)
            (2, 7),    # Then: go LEFT along the bottom walkway to x=2
            (2, 2),    # Finally: go UP to stairs at (2,2)
        ]
        
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
        
        # GUARD: Handle None map_id (happens during map buffer corruption)
        if map_id is None:
            # For exit goals, use DOWN as emergency fallback
            if goal in ['exit', (None, None)] or (isinstance(goal, str) and 'exit' in goal.lower()):
                return {'action': 'DOWN', 'reason': 'EMERGENCY: No map_id, generic DOWN to exit'}
            # For other goals, can't navigate without map context
            logger.warning(f"TerrainNav: map_id is None, cannot navigate to {goal}")
            return None
        
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
        
        # OPTIMIZATION 2: At goal or 1 tile away?
        # For house exits/stairs, need to STEP ON tile, not interact next to it!
        at_goal = abs(x - goal_x) <= 1 and abs(y - goal_y) <= 1
        exactly_at_goal = x == goal_x and y == goal_y
        
        if exactly_at_goal:
            # At exact position - interact
            self.last_pos = current_pos
            self.last_goal = goal
            self.last_direction = {'action': 'A', 'reason': f'At goal ({goal_x},{goal_y}) - interact!'}
            return self.last_direction
        elif at_goal:
            # 1 tile away - return direction to step onto goal tile
            dx = goal_x - x
            dy = goal_y - y
            if abs(dy) >= abs(dx):
                direction = 'UP' if dy < 0 else 'DOWN'
            else:
                direction = 'LEFT' if dx < 0 else 'RIGHT'
            self.last_pos = current_pos
            self.last_goal = goal
            self.last_direction = {'action': direction, 'reason': f'1 tile from ({goal_x},{goal_y}) - step onto it'}
            return self.last_direction
        
        # ROUTE 101 & ROUTE 103 & OLDALE TOWN: Skip simple path, use A* with ground truth metatile data
        # Route 101: Complex terrain with trees/ledges throughout - simple path gets stuck!
        # Route 103: Has ledges blocking path to rival - must use A* to go around
        # Oldale Town: Buildings block direct paths - must navigate around
        # Lab interiors: Complex furniture/equipment requires A* to navigate around
        skip_simple_path = (
            (map_id == 0x10) or  # Route 101 - ALWAYS use A* (ground truth terrain)
            (map_id == 0x11) or  # Oldale Town - buildings block paths
            (map_id == 0x12) or  # Route 103 - always use A* due to ledges
            (map_id in [0x020C, 0x104, 0x0])  # Lab interiors - complex obstacles
        )
        
        # ═══════════════════════════════════════════════════════════════════════════
        # ROUTE 101 SPECIAL HANDLING: Use waypoint-based navigation
        # A* can't find path across entire map due to 15x15 visibility and ledges
        # Use intermediate waypoints to navigate around obstacles
        # ═══════════════════════════════════════════════════════════════════════════
        if map_id == 0x10:  # Route 101
            # Determine direction: going north to Oldale or south to Littleroot?
            going_north = goal_y < y  # Goal has lower y = going UP = north
            
            result = self.get_direction_route101(x, y, goal, metatile_info, going_north)
            if result:
                self.last_pos = current_pos
                self.last_goal = goal
                self.last_direction = result
                return result
        
        # ═══════════════════════════════════════════════════════════════════════════
        # ROUTE 103 SPECIAL HANDLING: Use waypoint-based navigation to reach rival
        # Ledges block direct UP path, must snake around via RIGHT
        # ═══════════════════════════════════════════════════════════════════════════
        if map_id == 0x12:  # Route 103
            # Going north to rival (goal y < current y means going UP)
            going_north = goal_y < y
            
            result = self.get_direction_route103(x, y, goal, metatile_info, going_north)
            if result:
                self.last_pos = current_pos
                self.last_goal = goal
                self.last_direction = result
                return result
        
        # ═══════════════════════════════════════════════════════════════════════════
        # OLDALE TOWN SPECIAL HANDLING: Use waypoints to navigate around building
        # Building at (3-6, y=14) blocks direct path to PokeCenter at (6,17)
        # ═══════════════════════════════════════════════════════════════════════════
        if map_id == 0x11:  # Oldale Town
            # Check if goal is PokeCenter (6,17) or nearby
            if goal_y >= 16 and goal_y <= 18 and goal_x >= 4 and goal_x <= 8:
                result = self.get_direction_oldale(x, y, goal, metatile_info)
                if result:
                    self.last_pos = current_pos
                    self.last_goal = goal
                    self.last_direction = result
                    return result
        
        # ═══════════════════════════════════════════════════════════════════════════
        # PLAYER'S HOUSE 1F SPECIAL HANDLING: Waypoints to navigate around moving NPC
        # May's House 1F (0x102) or Brendan's House 1F (0x100) - same layout
        # Moving Pokemon/NPC blocks direct path to stairs at (2,2)
        # ═══════════════════════════════════════════════════════════════════════════
        if map_id in [0x102, 0x100]:  # May's or Brendan's House 1F
            # Check if goal is stairs at (2,2) or nearby
            if goal_y <= 3 and goal_x <= 3:
                result = self.get_direction_house_1f(x, y, goal, metatile_info)
                if result:
                    self.last_pos = current_pos
                    self.last_goal = goal
                    self.last_direction = result
                    return result
        
        # ═══════════════════════════════════════════════════════════════════════════
        # PLAYER'S/RIVAL'S HOUSE 2F SPECIAL HANDLING: Navigate to stairs to descend
        # May's House 2F (0x103) or Brendan's House 2F (0x101) - stairs at (2,2)
        # ═══════════════════════════════════════════════════════════════════════════
        if map_id in [0x103, 0x101]:  # May's or Brendan's House 2F
            # Check if goal is stairs at (2,2) or nearby
            if goal_y <= 3 and goal_x <= 3:
                result = self.get_direction_house_2f(x, y, goal, metatile_info)
                if result:
                    self.last_pos = current_pos
                    self.last_goal = goal
                    self.last_direction = result
                    return result
        
        # OPTIMIZATION 3: Try simple straight-line movement (UNLESS in ledge zone)
        if not skip_simple_path:
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
        
        # OPTIMIZATION 5: Bounded A* (max 300 nodes for complex ledge paths)
        self.cache_misses += 1
        path = self._bounded_astar(x, y, goal_x, goal_y, metatile_info, max_nodes=300)
        
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
        self.last_direction = self._heuristic_direction(x, y, goal_x, goal_y, metatile_info)
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
                    
                    # CRITICAL: Tile ID 1023 = out-of-bounds - skip!
                    tile_id = tile.get('id', 0)
                    if tile_id == 1023:
                        continue
                    
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
                    
                    # DEBUG: Log when we're considering moves (every 100 calls)
                    if hasattr(self, 'simple_dir_calls'):
                        self.simple_dir_calls += 1
                    else:
                        self.simple_dir_calls = 0
                    if self.simple_dir_calls % 100 == 0:
                        logger.info(f"🧪 SIMPLE_DIR [{direction}]: tile[{center + oy},{center + ox}] = {behavior}, passable={passable}, score={old_dist - new_dist}")

                    
                    # Valid move! Score by distance reduction
                    score = old_dist - new_dist
                    if score > best_score:
                        best_score = score
                        best_dir = direction
                except (IndexError, KeyError, TypeError):
                    continue
        
        if best_dir:
            return {'action': best_dir, 'reason': f'Simple path to ({goal_x},{goal_y})'}
        
        # DEBUG: Log when simple path fails (every 50 calls)
        if hasattr(self, 'simple_dir_calls'):
            self.simple_dir_calls += 1
        else:
            self.simple_dir_calls = 0
        if self.simple_dir_calls % 50 == 0:
            logger.info(f"🔍 SIMPLE_DIR FAILED at ({x},{y})->({goal_x},{goal_y}): No valid direction found, will try A*")
        
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
        
        def is_passable(px: int, py: int, from_direction: str = None) -> bool:
            # Convert to grid coordinates
            gx = center + (px - x)
            gy = center + (py - y)
            
            # Outside visible grid - assume passable
            if gx < 0 or gx >= 15 or gy < 0 or gy >= 15:
                return True
            
            try:
                tile = metatile_info[gy][gx]
                
                # CRITICAL: Tile ID 1023 (0x3FF) = out-of-bounds/unloaded - NEVER passable!
                tile_id = tile.get('id', 0)
                if tile_id == 1023:
                    return False
                
                behavior = tile.get('behavior', '')
                
                if 'IMPASSABLE' in behavior:
                    return False
                
                # === LEDGE HANDLING (ONE-WAY!) ===
                # JUMP_SOUTH means you can only LEAVE going south (DOWN)
                # Cannot ENTER from south (which would be moving UP)
                if 'JUMP' in behavior:
                    # JUMP_SOUTH: can only go DOWN (south), blocks UP (entering from south)
                    if 'SOUTH' in behavior:
                        # DOWN movement = entering from north = allowed (jumping down)
                        # UP movement = trying to climb = BLOCKED
                        return from_direction == 'DOWN'
                    elif 'NORTH' in behavior:
                        return from_direction == 'UP'
                    elif 'EAST' in behavior:
                        return from_direction == 'RIGHT'
                    elif 'WEST' in behavior:
                        return from_direction == 'LEFT'
                    else:
                        # Generic JUMP - assume south-facing
                        return from_direction == 'DOWN'
                
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
            
            # Goal reached? MUST be EXACTLY at goal, not 1 tile away!
            # The "1 tile away" logic is handled in get_direction() BEFORE calling A*
            if cx == goal_x and cy == goal_y:
                return path if path else ['A']
            
            # Explore neighbors - pass direction for one-way ledge handling
            for direction, (dx, dy) in [('UP', (0, -1)), ('DOWN', (0, 1)), 
                                         ('LEFT', (-1, 0)), ('RIGHT', (1, 0))]:
                nx, ny = cx + dx, cy + dy
                
                # Pass direction so ledges can be crossed correctly (DOWN=jump allowed)
                if (nx, ny) not in visited and is_passable(nx, ny, from_direction=direction):
                    new_g = g + 1
                    new_f = new_g + heuristic(nx, ny)
                    new_path = path + [direction]
                    heapq.heappush(open_set, (new_f, new_g, (nx, ny), new_path))

        
        # No path found within budget
        return None
    
    def _heuristic_direction(self, x: int, y: int, goal_x: int, goal_y: int, metatile_info: List[List[Dict]] = None) -> Dict:
        """Fallback: simple direction towards goal with terrain check."""
        dx = goal_x - x
        dy = goal_y - y
        
        # Order directions by preference towards goal
        directions = []
        if abs(dx) > abs(dy):
            directions = ['RIGHT' if dx > 0 else 'LEFT', 'UP' if dy < 0 else 'DOWN']
        else:
            directions = ['UP' if dy < 0 else 'DOWN', 'RIGHT' if dx > 0 else 'LEFT']
        
        # Check terrain if available
        if metatile_info and len(metatile_info) >= 15:
            center = 7
            dir_offsets = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}
            
            for dir in directions:
                ox, oy = dir_offsets[dir]
                try:
                    tile = metatile_info[center + oy][center + ox]
                    behavior = tile.get('behavior', '')
                    passable = tile.get('passable', True)
                    
                    # Skip ledges for non-DOWN moves, skip impassable
                    if 'JUMP' in behavior and dir != 'DOWN':
                        continue
                    if 'IMPASSABLE' in behavior or not passable:
                        continue
                    
                    return {'action': dir, 'reason': f'Heuristic with terrain (fallback)'}
                except (IndexError, KeyError, TypeError):
                    continue
        
        # Ultimate fallback without terrain check
        if abs(dx) > abs(dy):
            action = 'RIGHT' if dx > 0 else 'LEFT'
        else:
            action = 'DOWN' if dy > 0 else 'UP'
        
        return {'action': action, 'reason': 'Heuristic (no terrain data)'}
    
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
    
    def get_route101_next_waypoint(
        self, 
        x: int, 
        y: int, 
        going_north: bool = True
    ) -> Optional[Tuple[int, int]]:
        """
        Get the next waypoint for Route 101 navigation.
        
        Route 101 has complex terrain with ledges/trees that A* can't solve
        in one shot. This returns intermediate waypoints to navigate around.
        
        Args:
            x, y: Current position on Route 101
            going_north: True = heading to Oldale, False = heading to Littleroot
            
        Returns:
            (waypoint_x, waypoint_y) or None if at destination
        """
        waypoints = self.route101_waypoints_north if going_north else self.route101_waypoints_south
        
        # Find the first waypoint that's "ahead" of current position
        # For north: waypoint y should be < current y (or close but different x)
        # For south: waypoint y should be > current y
        
        for wx, wy in waypoints:
            if going_north:
                # Going north (UP): target waypoints with lower y (north of us)
                # CRITICAL: Skip waypoints that are SOUTH of current position!
                if wy > y:
                    # This waypoint is behind us (south), skip it
                    continue
                    
                if wy < y - 2:  # Waypoint is clearly north of us
                    return (wx, wy)
                elif wy <= y and abs(wy - y) <= 3 and abs(wx - x) > 2:
                    # Same y level or slightly north, different x = need to go around obstacle
                    # Only if waypoint is NOT south of us
                    return (wx, wy)
            else:
                # Going south (DOWN): target waypoints with higher y (south of us)
                if wy < y:
                    # This waypoint is behind us (north), skip it
                    continue
                    
                if wy > y + 2:
                    return (wx, wy)
                elif wy >= y and abs(wy - y) <= 3 and abs(wx - x) > 2:
                    return (wx, wy)
        
        # At or past last waypoint - return final destination
        if going_north:
            return (8, 0)   # Oldale entrance
        else:
            return (8, 28)  # Littleroot exit
    
    def get_direction_route101(
        self,
        x: int,
        y: int,
        goal: Union[str, Tuple[int, int]],
        metatile_info: List[List[Dict]],
        going_north: bool = True
    ) -> Optional[Dict]:
        """
        Special Route 101 navigation using waypoints.
        
        Instead of A* to distant goal, finds next waypoint and A* to that.
        """
        # Get next waypoint
        next_wp = self.get_route101_next_waypoint(x, y, going_north)
        
        if not next_wp:
            # At destination
            return {'action': 'A', 'reason': 'At Route 101 destination'}
        
        wp_x, wp_y = next_wp
        
        # Check if we're close to this waypoint
        dist_to_wp = abs(x - wp_x) + abs(y - wp_y)
        if dist_to_wp <= 2:
            # At waypoint, get next one
            # Temporarily move "past" this waypoint for next iteration
            if going_north and wp_y > 0:
                next_wp = self.get_route101_next_waypoint(wp_x, wp_y - 3, going_north)
            elif not going_north and wp_y < 28:
                next_wp = self.get_route101_next_waypoint(wp_x, wp_y + 3, going_north)
            
            if next_wp:
                wp_x, wp_y = next_wp
        
        # A* to waypoint (short distance, within 15x15 grid visibility)
        path = self._bounded_astar(x, y, wp_x, wp_y, metatile_info, max_nodes=150)
        
        if path:
            return {'action': path[0], 'reason': f'Route101 waypoint ({wp_x},{wp_y})'}
        
        # A* failed - use simple heuristic to waypoint
        return self._heuristic_direction(x, y, wp_x, wp_y, metatile_info)
    
    def get_route103_next_waypoint(
        self, 
        x: int, 
        y: int, 
        going_north: bool = True
    ) -> Optional[Tuple[int, int]]:
        """
        Get the next waypoint for Route 103 navigation.
        
        Route 103 has ledges blocking direct UP path, must snake around via RIGHT.
        FIXED: If agent at x >= 20 (east side), use recovery waypoints to go LEFT.
        
        Args:
            x, y: Current position on Route 103
            going_north: True = heading to rival, False = heading to Oldale
            
        Returns:
            (waypoint_x, waypoint_y) or None if at destination
        """
        # EAST RECOVERY: If agent is far east (x >= 18), use special recovery path
        # This guides agent LEFT back toward the rival area
        if going_north and x >= 18:
            waypoints = self.route103_east_recovery
            logger.info(f"🔧 ROUTE103 EAST RECOVERY: x={x} >= 18, using recovery waypoints")
        elif going_north:
            waypoints = self.route103_waypoints_north
        else:
            waypoints = self.route103_waypoints_south
        
        # Find the first waypoint that's "ahead" of current position
        for wx, wy in waypoints:
            if going_north:
                # Going north (to rival): target waypoints with lower y OR lower x (for east recovery)
                if x >= 18:
                    # EAST RECOVERY MODE: prioritize going LEFT (lower x)
                    if wx < x - 2:  # Waypoint is clearly west (LEFT) of us
                        return (wx, wy)
                    elif abs(wy - y) <= 3 and wx < x:
                        # Same y level, waypoint to the left
                        return (wx, wy)
                else:
                    # Normal north navigation
                    if wy > y:
                        # This waypoint is behind us (south), skip it
                        continue
                        
                    if wy < y - 2:  # Waypoint is clearly north of us
                        return (wx, wy)
                    elif abs(wy - y) <= 3 and abs(wx - x) > 2:
                        # Same y level but different x = need to go around obstacle
                        return (wx, wy)
            else:
                # Going south (to Oldale): target waypoints with higher y
                if wy < y:
                    # This waypoint is behind us (north), skip it
                    continue
                    
                if wy > y + 2:
                    return (wx, wy)
                elif abs(wy - y) <= 3 and abs(wx - x) > 2:
                    return (wx, wy)
        
        # At or past last waypoint - return final destination
        if going_north:
            return (10, 3)   # Rival area
        else:
            return (5, 15)   # Oldale exit
    
    def get_direction_route103(
        self,
        x: int,
        y: int,
        goal: Union[str, Tuple[int, int]],
        metatile_info: List[List[Dict]],
        going_north: bool = True
    ) -> Optional[Dict]:
        """
        Special Route 103 navigation using waypoints.
        
        Ledges block direct UP path, so navigate via intermediate waypoints.
        """
        # Get next waypoint
        next_wp = self.get_route103_next_waypoint(x, y, going_north)
        
        if not next_wp:
            # At destination
            return {'action': 'A', 'reason': 'At Route 103 destination (rival area)'}
        
        wp_x, wp_y = next_wp
        
        # Check if we're close to this waypoint
        dist_to_wp = abs(x - wp_x) + abs(y - wp_y)
        if dist_to_wp <= 2:
            # At waypoint, get next one
            if going_north and wp_y > 0:
                next_wp = self.get_route103_next_waypoint(wp_x, wp_y - 3, going_north)
            elif not going_north and wp_y < 15:
                next_wp = self.get_route103_next_waypoint(wp_x, wp_y + 3, going_north)
            
            if next_wp:
                wp_x, wp_y = next_wp
        
        # Log waypoint navigation (every 20 calls)
        if not hasattr(self, 'route103_log_count'):
            self.route103_log_count = 0
        self.route103_log_count += 1
        if self.route103_log_count % 20 == 0:
            logger.info(f"🗺️ ROUTE103 WAYPOINT: ({x},{y}) -> ({wp_x},{wp_y})")
        
        # A* to waypoint (short distance, within 15x15 grid visibility)
        path = self._bounded_astar(x, y, wp_x, wp_y, metatile_info, max_nodes=150)
        
        if path:
            return {'action': path[0], 'reason': f'Route103 waypoint ({wp_x},{wp_y})'}
        
        # A* failed - use simple heuristic to waypoint
        return self._heuristic_direction(x, y, wp_x, wp_y, metatile_info)
    
    def get_direction_oldale(
        self,
        x: int,
        y: int,
        goal: Union[str, Tuple[int, int]],
        metatile_info: List[List[Dict]]
    ) -> Optional[Dict]:
        """
        Oldale Town special navigation around building to PokeCenter.
        
        Building at (3-6, y=14) blocks direct path. Must go:
        1. LEFT to x=2 (past building)
        2. DOWN to y=17 (same row as PokeCenter)
        3. RIGHT to x=6 (PokeCenter entrance)
        """
        # Get next waypoint based on current position
        wp_x, wp_y = self._get_oldale_next_waypoint(x, y)
        
        # Log for debugging
        if not hasattr(self, 'oldale_log_count'):
            self.oldale_log_count = 0
        self.oldale_log_count += 1
        if self.oldale_log_count % 10 == 0:
            logger.info(f"🏥 OLDALE WAYPOINT NAV: ({x},{y}) -> waypoint ({wp_x},{wp_y})")
        
        # Simple directional logic for each phase
        # KEY INSIGHT: Building at (3-6, y=14) - if y > 14, agent is PAST building!
        
        # SMART ROUTING: If y > 14 (past building), go directly RIGHT/DOWN
        if y > 14:
            # Already past the building, take direct route
            if y < 17:
                return {'action': 'DOWN', 'reason': f'Oldale: past building, go DOWN to PC (y={y}->17)'}
            if x < 6:
                return {'action': 'RIGHT', 'reason': f'Oldale: past building, go RIGHT to PC (x={x}->6)'}
            if x > 6:
                return {'action': 'LEFT', 'reason': f'Oldale: past building, go LEFT to PC (x={x}->6)'}
        
        # Phase 1: Go LEFT until x <= 2 (only if still at y <= 14, blocked by building)
        if x > 2 and y <= 14:
            return {'action': 'LEFT', 'reason': f'Oldale: go LEFT past building (x={x}->2, y={y})'}
        
        # If x > 2 but y <= 14, try DOWN first to get past building row
        if y <= 14:
            return {'action': 'DOWN', 'reason': f'Oldale: go DOWN past building row (y={y}->15)'}
        
        # Phase 2: Go DOWN until y >= 17 (already past building)
        if y < 17:
            return {'action': 'DOWN', 'reason': f'Oldale: go DOWN to PC row (y={y}->17)'}
        
        # Phase 3: Go RIGHT until x >= 6
        if x < 6:
            return {'action': 'RIGHT', 'reason': f'Oldale: go RIGHT to PC (x={x}->6)'}
        
        # At PokeCenter entrance - interact!
        if abs(x - 6) <= 1 and abs(y - 17) <= 1:
            return {'action': 'UP', 'reason': 'At PokeCenter door - ENTER!'}
        
        # Fallback to A* for final approach
        path = self._bounded_astar(x, y, 6, 17, metatile_info, max_nodes=50)
        if path:
            return {'action': path[0], 'reason': f'Oldale A* to PC ({6},{17})'}
        
        return self._heuristic_direction(x, y, 6, 17, metatile_info)
    
    def _get_oldale_next_waypoint(self, x: int, y: int) -> Tuple[int, int]:
        """Get next waypoint for Oldale Town PokeCenter navigation."""
        # Phase 1: Go LEFT past building (until x <= 2)
        if x > 2:
            return (2, y)  # Same row, just go LEFT
        
        # Phase 2: Go DOWN to PokeCenter row (y=17)
        if y < 17:
            return (x, 17)  # Same column, go DOWN
        
        # Phase 3: Go RIGHT to PokeCenter entrance (x=6)
        return (6, 17)  # Final destination
    
    def get_direction_house_1f(
        self,
        x: int,
        y: int,
        goal: Union[str, Tuple[int, int]],
        metatile_info: List[List[Dict]]
    ) -> Optional[Dict]:
        """
        Special House 1F navigation using waypoints to reach stairs.
        
        Moving Pokemon/NPC blocks direct path to stairs at (2,2).
        L-shaped path: go DOWN to bottom walkway (y=7), LEFT to x=2, then UP to stairs.
        """
        # Log for debugging
        if not hasattr(self, 'house_log_count'):
            self.house_log_count = 0
        self.house_log_count += 1
        if self.house_log_count % 10 == 0:
            logger.info(f"🏠 HOUSE 1F NAV: ({x},{y}) -> stairs (2,2)")
        
        # At or near stairs - go up to enter
        if x <= 3 and y <= 3:
            if x < 2:
                return {'action': 'RIGHT', 'reason': f'House1F: At stairs area, go RIGHT (x={x}->2)'}
            if x > 2:
                return {'action': 'LEFT', 'reason': f'House1F: At stairs area, go LEFT (x={x}->2)'}
            if y > 2:
                return {'action': 'UP', 'reason': f'House1F: At stairs, go UP to 2F (y={y}->2)'}
            # At exact stairs position
            return {'action': 'UP', 'reason': 'House1F: At stairs (2,2) - go up to 2F!'}
        
        # Phase 1: If not at bottom walkway (y < 7), go DOWN first
        # NPC moves in middle area, so we go to the safe bottom walkway
        if y < 7:
            return {'action': 'DOWN', 'reason': f'House1F: go DOWN to bottom walkway (y={y}->7)'}
        
        # Phase 2: Go LEFT along bottom walkway until x <= 2
        if x > 2:
            return {'action': 'LEFT', 'reason': f'House1F: go LEFT along bottom (x={x}->2)'}
        
        # Phase 3: Go UP to stairs at (2,2)
        if y > 2:
            return {'action': 'UP', 'reason': f'House1F: go UP to stairs (y={y}->2)'}
        
        # Should be at stairs now
        return {'action': 'UP', 'reason': 'House1F: at stairs position, go UP to 2F'}
    
    def get_direction_house_2f(
        self,
        x: int,
        y: int,
        goal: Union[str, Tuple[int, int]],
        metatile_info: List[List[Dict]]
    ) -> Optional[Dict]:
        """
        Special House 2F navigation to reach stairs and go DOWN to 1F.
        
        Stairs at (2,2) - agent needs to walk onto stairs tile to descend.
        In Pokemon, stairs are usually warp tiles that trigger on step-on.
        """
        # Log for debugging
        if not hasattr(self, 'house2f_log_count'):
            self.house2f_log_count = 0
        self.house2f_log_count += 1
        if self.house2f_log_count % 10 == 0:
            logger.info(f"🏠 HOUSE 2F NAV: ({x},{y}) -> stairs (1,1) [visual map]")
        
        # At or very near stairs - step onto it (walk onto warp tile)
        if x == 1 and y == 1:
            # At exact stairs position - press DOWN to descend (or step off and back on)
            return {'action': 'DOWN', 'reason': 'House2F: At stairs (1,1) - go DOWN to 1F!'}
        
        # Very close to stairs - move to exact position
        if x <= 3 and y <= 3:
            if x < 1:
                return {'action': 'RIGHT', 'reason': f'House2F: Near stairs, go RIGHT (x={x}->1)'}
            if x > 1:
                return {'action': 'LEFT', 'reason': f'House2F: Near stairs, go LEFT (x={x}->1)'}
            if y > 1:
                return {'action': 'UP', 'reason': f'House2F: Near stairs, go UP (y={y}->1)'}
            if y < 1:
                return {'action': 'DOWN', 'reason': f'House2F: Near stairs, go DOWN (y={y}->1)'}
        
        # Navigate to stairs at (1,1) - go LEFT and UP
        if x > 1:
            return {'action': 'LEFT', 'reason': f'House2F: go LEFT to stairs (x={x}->1)'}
        
        if y > 1:
            return {'action': 'UP', 'reason': f'House2F: go UP to stairs (y={y}->1)'}
        
        # Should be at or near stairs
        return {'action': 'LEFT', 'reason': 'House2F: navigate to stairs (1,1)'}
    
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
