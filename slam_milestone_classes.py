"""
SLAM-like Spatial Mapper and Milestone Feedback System
For path planning and server milestone integration
"""

import requests
from heapq import heappush, heappop


class SLAMSpatialMapper:
    """
    SLAM-inspired spatial mapper for path planning (Simultaneous Localization And Mapping).
    
    Features:
    • Occupancy grid: Build map from visited tiles (walkable/blocked)
    • Path planning: A* from current position to milestone target
    • Spatial memory: Remember visited locations (exploration bonus)
    • Landmark detection: Key locations (gyms, towns, NPCs) for waypoints
    """
    def __init__(self, map_size=256):
        self.map_size = map_size
        # Occupancy grid: -1=unknown, 0=free, 1=blocked, 2=landmark
        self.occupancy_grid = {}  # (map_id, x, y) -> occupancy_value
        self.visited_positions = set()  # (map_id, x, y)
        self.landmarks = {}  # (map_id, x, y) -> landmark_type (GYM, LAB, NPC)
        
        # Path planning state
        self.current_path = []  # List of (x, y) waypoints
        self.path_target = None  # (map_id, x, y)
        
        # Exploration metrics
        self.exploration_count = {}  # (map_id, x, y) -> visit_count
    
    def update_position(self, map_id, x, y, tile_info):
        """Update occupancy grid from current observation"""
        key = (map_id, x, y)
        self.visited_positions.add(key)
        self.exploration_count[key] = self.exploration_count.get(key, 0) + 1
        
        # Update occupancy based on tile behavior
        if tile_info.get('is_blocked', False):
            self.occupancy_grid[key] = 1  # Blocked
        elif tile_info.get('is_grass', False) or tile_info.get('is_door', False):
            self.occupancy_grid[key] = 0  # Free/walkable
        else:
            self.occupancy_grid[key] = 0  # Assume free until proven blocked
    
    def add_landmark(self, map_id, x, y, landmark_type):
        """Register important landmark (gym, lab, NPC)"""
        key = (map_id, x, y)
        self.landmarks[key] = landmark_type
        self.occupancy_grid[key] = 2  # Mark as landmark
    
    def update_from_server_map(self, map_id, metatile_info):
        """
        Ingest complete terrain map from server (15×15 grid with ground-truth walkability).
        This provides perfect spatial awareness for path planning!
        
        Args:
            map_id: Current map ID
            metatile_info: 2D list from server's /state endpoint
                          Each tile has: passable (bool), behavior (str), collision, elevation
        """
        if not metatile_info:
            return
        
        for y, row in enumerate(metatile_info):
            for x, tile_data in enumerate(row):
                key = (map_id, x, y)
                
                # Use server's ground-truth passable flag
                is_walkable = tile_data.get('passable', True)
                behavior = tile_data.get('behavior', '')
                
                # Update occupancy grid
                if not is_walkable:
                    self.occupancy_grid[key] = 1  # Blocked
                else:
                    self.occupancy_grid[key] = 0  # Walkable
                
                # Detect special tiles (doors/warps = landmarks for navigation)
                if 'WARP' in behavior or 'DOOR' in behavior:
                    self.add_landmark(map_id, x, y, 'EXIT_DOOR')
                elif 'STAIRS' in behavior:
                    self.add_landmark(map_id, x, y, 'STAIRS')
                
                # Store additional info for advanced path planning
                if not hasattr(self, 'tile_details'):
                    self.tile_details = {}
                self.tile_details[key] = {
                    'collision': tile_data.get('collision', 0),
                    'elevation': tile_data.get('elevation', 0),
                    'behavior': behavior,
                    'encounter': tile_data.get('encounter_possible', False)
                }
    
    def plan_path(self, start_map, start_x, start_y, target_map, target_x, target_y):
        """
        A* path planning from start to target.
        Returns: List of (x, y) waypoints, or None if unreachable
        """
        if start_map != target_map:
            # Need map transition - find door/warp to target map
            return None  # TODO: Multi-map planning
        
        # Simple A* within same map
        open_set = [(0, start_x, start_y)]
        came_from = {}
        g_score = {(start_x, start_y): 0}
        
        def heuristic(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance
        
        while open_set:
            _, current_x, current_y = heappop(open_set)
            
            if (current_x, current_y) == (target_x, target_y):
                # Reconstruct path
                path = [(current_x, current_y)]
                while (current_x, current_y) in came_from:
                    current_x, current_y = came_from[(current_x, current_y)]
                    path.append((current_x, current_y))
                return list(reversed(path))
            
            # Check 4-connected neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                neighbor_key = (start_map, neighbor_x, neighbor_y)
                
                # Check if blocked
                if self.occupancy_grid.get(neighbor_key, -1) == 1:
                    continue  # Blocked tile
                
                tentative_g = g_score[(current_x, current_y)] + 1
                
                if (neighbor_x, neighbor_y) not in g_score or tentative_g < g_score[(neighbor_x, neighbor_y)]:
                    came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                    g_score[(neighbor_x, neighbor_y)] = tentative_g
                    f_score = tentative_g + heuristic(neighbor_x, neighbor_y, target_x, target_y)
                    heappush(open_set, (f_score, neighbor_x, neighbor_y))
        
        return None  # No path found
    
    def get_navigation_hints(self, current_map, current_x, current_y):
        """
        Get 64-dim navigation hints for SNN path planning.
        Returns: direction preferences, exploration bonus, landmark proximity, path progress
        
        KEY INSIGHT: This is the "geometry brain" that bridges dialogue → SLAM → RL:
        - Dialogue sets WHAT to do (quest objective)
        - SLAM determines WHERE and HOW to get there (path waypoints)
        - SNN remembers WHY and FOR HOW LONG (temporal context)
        - RL executes actual movement (fine motor control)
        """
        hints = [0.0] * 64
        
        # [0-3]: Direction preferences based on current path (A* guidance)
        if self.current_path and len(self.current_path) > 1:
            next_x, next_y = self.current_path[1]
            dx, dy = next_x - current_x, next_y - current_y
            if dx > 0: hints[0] = 1.0  # East
            elif dx < 0: hints[1] = 1.0  # West
            if dy > 0: hints[2] = 1.0  # South
            elif dy < 0: hints[3] = 1.0  # North
        
        # [4-7]: Exploration bonus (prefer unvisited areas)
        # Fallback when no path available
        for i, (dx, dy) in enumerate([(1, 0), (-1, 0), (0, 1), (0, -1)]):
            neighbor_key = (current_map, current_x + dx, current_y + dy)
            visits = self.exploration_count.get(neighbor_key, 0)
            hints[4 + i] = max(0.0, 1.0 - visits / 10.0)  # Decay with visits
        
        # [8]: Landmark proximity (gym, lab, NPC within 10 tiles)
        nearest_landmark_dist = float('inf')
        for (lmap, lx, ly), ltype in self.landmarks.items():
            if lmap == current_map:
                dist = abs(lx - current_x) + abs(ly - current_y)
                if dist < nearest_landmark_dist:
                    nearest_landmark_dist = dist
        
        if nearest_landmark_dist < 10:
            hints[8] = 1.0 - nearest_landmark_dist / 10.0  # Proximity signal
        
        # [9-11]: Path progress (SNN temporal reasoning)
        if self.current_path:
            total_waypoints = len(self.current_path)
            # Find closest waypoint
            min_dist = float('inf')
            closest_idx = 0
            for i, (wx, wy) in enumerate(self.current_path):
                dist = abs(wx - current_x) + abs(wy - current_y)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            hints[9] = closest_idx / max(total_waypoints - 1, 1)  # Progress ratio [0-1]
            hints[10] = min_dist / 10.0  # Distance to next waypoint (normalized)
            hints[11] = 1.0 if min_dist > 5 else 0.0  # Deviation flag (stuck/lost)
        
        # [12-19]: Multi-step quest chain progress (temporal planning)
        # Reserved for quest chain like: FIND_MAY → BATTLE_MAY → RETURN_TO_BIRCH
        # This complements milestone tracking with sub-quest sequences
        
        # [20-27]: Landmark type signals (what kind of goal)
        # GYM=1, LAB=2, NPC=3, ROUTE=4, TOWN=5, etc.
        # Helps SNN understand goal semantics
        
        # [28-63]: Reserved for route hints, terrain preferences, temporal patterns
        
        return hints
    
    def get_exploration_bonus(self, map_id, x, y):
        """Reward for visiting new tiles (novelty exploration)"""
        key = (map_id, x, y)
        visits = self.exploration_count.get(key, 0)
        if visits == 0:
            return 1.0  # First visit bonus
        elif visits < 5:
            return 0.2  # Revisit small bonus
        return 0.0  # Over-visited
    
    def get_path_following_reward(self, current_map, current_x, current_y):
        """
        Reward for following A* path toward goal.
        Returns: (reward, status) where status is 'on_path', 'near_path', or 'lost'
        
        This implements: "A* path gives advice, not hard-coded script"
        - RL still handles fine motor control and local avoidance
        - Path following is rewarded but not enforced
        """
        if not self.current_path:
            return 0.0, 'no_path'
        
        # Find distance to closest waypoint
        min_dist = float('inf')
        for wx, wy in self.current_path:
            dist = abs(wx - current_x) + abs(wy - current_y)
            if dist < min_dist:
                min_dist = dist
        
        # Reward structure:
        # On waypoint (dist=0): +2.0
        # Near path (dist<3): +1.0
        # Off path (dist>5): 0.0
        # Lost (dist>10): -0.5 (small penalty, still allows exploration)
        if min_dist == 0:
            return 2.0, 'on_path'
        elif min_dist < 3:
            return 1.0, 'near_path'
        elif min_dist < 10:
            return 0.0, 'off_path'
        else:
            return -0.5, 'lost'  # Small penalty, not harsh
    
    def is_geometrically_consistent(self):
        """
        Check if quest is geometrically reachable.
        Returns: (bool, reason)
        
        This is the consistency check mentioned:
        'a boolean "quest is geometrically consistent" (reachable, map known)'
        """
        if not self.path_target:
            return False, 'No target set'
        
        target_map, target_x, target_y = self.path_target
        
        # Check if target location is in known maps
        target_key = (target_map, target_x, target_y)
        if target_key not in self.occupancy_grid:
            return False, f'Target map {target_map:04X} unknown'
        
        # Check if target is blocked
        if self.occupancy_grid[target_key] == 1:
            return False, 'Target location blocked'
        
        # Check if path exists
        if not self.current_path:
            return False, 'No path found to target'
        
        return True, 'Path exists and target reachable'


class MilestoneFeedbackSystem:
    """
    Query server /milestones endpoint to provide feedback to SNN/VLM reasoning.
    
    Purpose:
    • Track what's been done: completed milestones → memory
    • Plan what's needed: next milestone → goal for SNN
    • Spatial guidance: milestone → target location for SLAM path planning
    """
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url
        self.completed_milestones = set()
        self.current_milestone = None
        self.next_milestone_target = None  # (map_id, x, y, description)
        
        # Milestone → Location mapping (for path planning)
        self.milestone_locations = {
            'RIVAL_BEDROOM': (0x101, 3, 1, "May's bedroom 2F"),  # Brendan's house
            'ROUTE_101': (0x10, 12, 25, "Route 101 - Birch location"),
            'BIRCH_LAB_VISITED': (0x104, 6, 12, "Birch Lab"),
            'OLDALE_TOWN': (0x01, 10, 10, "Oldale Town center"),
            'ROUTE_103': (0x12, 5, 45, "Route 103 - rival battle"),
            'PETALBURG_CITY': (0x02, 12, 18, "Petalburg City"),
            'RUSTBORO_CITY': (0x07, 15, 20, "Rustboro City"),
            'RUSTBORO_GYM_ENTERED': (0x0D, 5, 14, "Rustboro Gym entrance"),
        }
    
    def update_from_server(self):
        """Query server for milestone status"""
        try:
            response = requests.get(f"{self.server_url}/milestones", timeout=1)
            if response.status_code == 200:
                data = response.json()
                # Update completed milestones
                for milestone, completed in data.items():
                    if completed and isinstance(completed, bool):
                        self.completed_milestones.add(milestone)
        except:
            pass  # Server unavailable
    
    def get_next_milestone_target(self, milestone_sequence):
        """
        Find next uncompleted milestone and return target location.
        Returns: (map_id, x, y, description) or None
        """
        for milestone in milestone_sequence:
            if milestone not in self.completed_milestones:
                self.current_milestone = milestone
                self.next_milestone_target = self.milestone_locations.get(milestone)
                return self.next_milestone_target
        return None
    
    def get_progress_encoding(self):
        """
        Encode milestone progress for SNN reasoning (32-dim).
        Returns: [completion_ratio, current_phase, milestones_completed, ...]
        """
        encoding = [0.0] * 32
        
        total_milestones = 35  # From comprehensive milestone list
        completed_count = len(self.completed_milestones)
        
        encoding[0] = completed_count / total_milestones  # Progress ratio
        encoding[1] = min(completed_count / 5, 7.0) / 7.0  # Phase (0-7 normalized)
        encoding[2] = 1.0 if 'STARTER_CHOSEN' in self.completed_milestones else 0.0
        encoding[3] = 1.0 if 'RECEIVED_POKEDEX' in self.completed_milestones else 0.0
        encoding[4] = 1.0 if 'FIRST_GYM_COMPLETE' in self.completed_milestones else 0.0
        
        # [5-31]: Reserved for specific milestone flags
        
        return encoding
