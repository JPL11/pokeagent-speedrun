#!/usr/bin/env python3
"""
Analyze why agent is stuck in Birch's Lab
Monitors training logs and diagnoses issues
"""

import re
import time
from collections import Counter, defaultdict

def analyze_log(log_file='training_optimized_512.log'):
    """Analyze training log to diagnose stuck behavior"""
    
    print("=" * 80)
    print("🔍 ANALYZING: Why is agent stuck in Birch's Lab?")
    print("=" * 80)
    
    # Track locations visited
    locations = Counter()
    lab_coordinates = []
    vlm_responses = []
    parsing_errors = []
    action_attempts = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Track VLM calls
            if 'Calling VLM oracle at' in line:
                match = re.search(r'at \((\d+),(\d+)\) map (\w+)', line)
                if match:
                    x, y, map_id = match.groups()
                    if map_id == '0104':  # Birch's Lab
                        lab_coordinates.append((int(x), int(y)))
                        locations[f"LAB ({x},{y})"] += 1
            
            # Track VLM responses
            if 'VLM SERVER-AWARE ANALYSIS' in line or 'VLM returned guidance' in line:
                # Get next few lines
                pass
            
            # Track parsing errors
            if "'NoneType' object has no attribute 'upper'" in line:
                parsing_errors.append(line.strip())
            
            # Track action attempts
            if 'Agent followed VLM advice:' in line:
                match = re.search(r'advice: (\w+)', line)
                if match:
                    action_attempts.append(match.group(1))
    
    # Analysis
    print(f"\n📍 LOCATION ANALYSIS:")
    print(f"   Total VLM calls in Lab (0x0104): {sum(1 for k in locations if 'LAB' in k)}")
    print(f"   Unique Lab coordinates visited: {len(set(lab_coordinates))}")
    print(f"   Most visited spots:")
    for loc, count in locations.most_common(10):
        print(f"      {loc}: {count} times")
    
    print(f"\n❌ PARSING ERRORS:")
    print(f"   Total NoneType errors: {len(parsing_errors)}")
    if parsing_errors:
        print(f"   Last error: {parsing_errors[-1]}")
    
    print(f"\n🎮 ACTION ATTEMPTS:")
    if action_attempts:
        action_counts = Counter(action_attempts)
        for action, count in action_counts.most_common():
            print(f"      {action}: {count} times")
    else:
        print(f"      ⚠️ NO ACTIONS FOLLOWED! VLM suggestions not being parsed correctly")
    
    # Lab coordinate analysis
    if lab_coordinates:
        min_x = min(x for x, y in lab_coordinates)
        max_x = max(x for x, y in lab_coordinates)
        min_y = min(y for y, y in lab_coordinates)
        max_y = max(y for y, y in lab_coordinates)
        
        print(f"\n🗺️ LAB EXPLORATION:")
        print(f"   X range: {min_x} to {max_x} (width: {max_x - min_x + 1})")
        print(f"   Y range: {min_y} to {max_y} (height: {max_y - min_y + 1})")
        print(f"   Lab exit is at bottom (y=9), agent needs to move DOWN/SOUTH")
    
    # Read last few VLM responses
    print(f"\n💬 LAST VLM RESPONSES:")
    with open(log_file, 'r') as f:
        content = f.read()
        # Find last VLM responses
        pattern = r'VLM SERVER-AWARE ANALYSIS.*?\n(.*?)(?=\n2025-|\Z)'
        matches = list(re.finditer(pattern, content, re.DOTALL))
        for match in matches[-3:]:
            response = match.group(1).strip()[:200]
            print(f"   {response}...")
            # Check if ACTION: present
            if 'ACTION:' in response.upper():
                print(f"      ✅ Has ACTION directive")
            else:
                print(f"      ❌ NO ACTION DIRECTIVE - VLM giving advice not commands!")
    
    return {
        'lab_visits': sum(1 for k in locations if 'LAB' in k),
        'parsing_errors': len(parsing_errors),
        'unique_coords': len(set(lab_coordinates)),
        'actions_followed': len(action_attempts)
    }


def diagnose_issues(stats):
    """Provide diagnosis and recommendations"""
    print("\n" + "=" * 80)
    print("🔬 DIAGNOSIS")
    print("=" * 80)
    
    issues = []
    
    if stats['parsing_errors'] > 0:
        issues.append("❌ VLM response parsing failures")
        print("\n1️⃣ **VLM RESPONSE PARSING ISSUE**")
        print("   Problem: VLM returning descriptive advice instead of ACTION: format")
        print("   Example: 'you need to move north through the tall grass'")
        print("   Expected: 'ACTION: DOWN' (to exit lab)")
        print("   Fix: Improve prompt to enforce ACTION: format or add fallback parsing")
    
    if stats['actions_followed'] == 0:
        issues.append("❌ VLM suggestions not being used")
        print("\n2️⃣ **VLM SUGGESTIONS NOT EXTRACTED**")
        print("   Problem: parse_action_from_response() returns None")
        print("   Reason: VLM not using 'ACTION: X' format in response")
        print("   Fix: Add regex fallback for natural language → action mapping")
    
    if stats['lab_visits'] > 50:
        issues.append("🔄 Stuck in Lab loop")
        print("\n3️⃣ **SPATIAL REASONING ISSUE**")
        print("   Problem: Agent doesn't know lab exit is DOWN/SOUTH")
        print("   Reason: High-level VLM says 'go north to Route 101' but forgets")
        print("           that FIRST you must exit lab (which requires going DOWN)")
        print("   Fix: Add spatial-temporal reasoning:")
        print("        • RGB-based door detection (exit door at bottom)")
        print("        • SLAM should track 'inside building' vs 'outside'")
        print("        • SNN temporal memory: 'I need to exit BEFORE going north'")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDED IMPROVEMENTS")
    print("=" * 80)
    
    print("\n1. **Fix VLM Action Parsing** (IMMEDIATE):")
    print("   - Add fallback regex: 'move|go|head (north|south|east|west|up|down)'")
    print("   - Map 'exit' → DOWN, 'enter' → A, 'talk' → A")
    print("   - Enforce prompt format with explicit examples")
    
    print("\n2. **Enhance Spatial Navigation** (SHORT-TERM):")
    print("   - RGB door detection: Feed to SLAM as 'exit points'")
    print("   - SLAM occupancy: Mark 'indoor' vs 'outdoor' zones")
    print("   - Milestone micro-steps: 'EXIT_LAB' before 'GO_TO_ROUTE_101'")
    
    print("\n3. **SNN Temporal Reasoning** (MEDIUM-TERM):")
    print("   - Quest chain: ['EXIT_LAB', 'GO_NORTH', 'ENTER_TALL_GRASS']")
    print("   - Path progress: Track if stuck at same coordinates")
    print("   - Memory of 'last exit direction' to avoid re-entering")
    
    print("\n4. **SLAM-RGB Integration** (YOUR SUGGESTION):")
    print("   - RGB detects doors/exits → SLAM marks as waypoints")
    print("   - SLAM path planning: 'To go north, first find exit (door), then navigate'")
    print("   - Feedback to SNN: 'Indoor=find_exit, Outdoor=follow_directions'")


if __name__ == '__main__':
    import sys
    log_file = sys.argv[1] if len(sys.argv) > 1 else 'training_optimized_512.log'
    
    stats = analyze_log(log_file)
    diagnose_issues(stats)
    
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Lab visits: {stats['lab_visits']}")
    print(f"Parsing errors: {stats['parsing_errors']}")
    print(f"Unique coords explored: {stats['unique_coords']}")
    print(f"Actions successfully followed: {stats['actions_followed']}")
