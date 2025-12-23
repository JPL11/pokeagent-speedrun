#!/usr/bin/env python3
"""
Real-time training dashboard - shows live metrics
"""

import requests
import time
import os
from collections import deque

def clear_screen():
    os.system('clear' if os.name != 'nt' else 'cls')

def get_training_stats():
    """Extract current training stats from log"""
    stats = {
        'current_step': 'N/A',
        'total_steps': '2,000,000',
        'progress_pct': 0.0,
        'recent_reward': 'N/A',
        'milestones': '0/?',
        'position': 'N/A',
        'map': 'N/A',
        'vlm_calls': 0,
        'episodic_memories': 0,
        'battle_events': 0,
        'log_file': 'N/A'
    }
    
    import glob
    
    # PRIORITY 1: Check for 'latest_training.log' symlink
    if os.path.exists('latest_training.log'):
        stats['log_file'] = 'latest_training.log'
        log_file = 'latest_training.log'
    else:
        # Fallback to finding newest file
        log_files = glob.glob('training_logs/training_*.log') + \
                    glob.glob('training_*.log') + \
                    (['training.log'] if os.path.exists('training.log') else [])
        
        if not log_files:
            stats['error'] = 'No training log file found'
            return stats
        
        # Get the newest log file
        log_file = max(log_files, key=os.path.getmtime)
        stats['log_file'] = log_file
    
    try:
        with open(log_file, 'r') as f:
            lines = deque(f, maxlen=1000)  # Last 1000 lines
            
            import re
            
            # Find current step (try multiple patterns)
            max_step = 0
            for line in lines:
                # Pattern 1: "Step 100:" or "🔍 Step 100:" or "Step 100 |" or "Step 100)"
                if match := re.search(r'Step\s+(\d+)[:\|\)]', line, re.IGNORECASE):
                    step = int(match.group(1))
                    max_step = max(max_step, step)
                # Pattern 2: "total_timesteps | 64"
                elif 'total_timesteps' in line:
                    if match := re.search(r'total_timesteps\s+\|\s+(\d+)', line):
                        step = int(match.group(1))
                        max_step = max(max_step, step)
                # Pattern 3: "step 100," (lowercase, in parentheses)
                elif match := re.search(r'step\s+(\d+)[,\s]', line, re.IGNORECASE):
                    step = int(match.group(1))
                    max_step = max(max_step, step)
            
            if max_step > 0:
                stats['current_step'] = f"{max_step:,}"
                stats['progress_pct'] = (max_step / 2000000) * 100
            
            # Recent reward
            for line in reversed(lines):
                if 'ep_rew_mean' in line:
                    # Handle both standard "ep_rew_mean: 10" and table "| ep_rew_mean | 10 |" formats
                    if match := re.search(r'ep_rew_mean\s*[|:]\s*([-0-9.]+)', line):
                        try:
                            stats['average_reward'] = float(match.group(1))
                            break
                        except ValueError:
                            continue
            stats['vlm_calls'] = sum(1 for l in lines if 'Calling VLM oracle' in l)
            
            # Count episodic memory stores
            stats['episodic_memories'] = sum(1 for l in lines if 'store_navigation_experience' in l or 'Episodic' in l)
            
            # Count battle events
            stats['battle_events'] = sum(1 for l in lines if 'battle' in l.lower() and ('won' in l or 'lost' in l or 'event' in l))
            
            # Count stuck detections
            stats['stuck_count'] = sum(1 for l in lines if '[STUCK]' in l)
            
            # Get FPS (from stable-baselines3 output)
            for line in reversed(lines):
                if 'fps' in line:
                    if match := re.search(r'fps\s+\|\s+([\d.]+)', line):
                        stats['fps'] = match.group(1)
                        break
            
            # Get k-NN confidence (if logged)
            confidences = []
            for line in lines:
                if 'k-NN confidence' in line or 'confidence=' in line:
                    if match := re.search(r'confidence[=:]\s*([\d.]+)', line):
                        confidences.append(float(match.group(1)))
            if confidences:
                stats['knn_confidence'] = f"{sum(confidences)/len(confidences):.3f}"
            
            # Calculate uptime from file modification time
            import time as time_module
            file_age = time_module.time() - os.path.getmtime(log_file)
            stats['uptime'] = file_age
    
    except Exception as e:
        stats['error'] = str(e)
    
    # Get live state from server
    try:
        response = requests.get('http://localhost:8000/state', timeout=1)
        if response.status_code == 200:
            state = response.json()
            player = state.get('player', {})
            pos = player.get('position', {})
            stats['position'] = f"({pos.get('x', '?')},{pos.get('y', '?')})"
            stats['map'] = player.get('location', 'UNKNOWN')[:40]
            
            # Get party info
            game_data = state.get('game', {})
            party = game_data.get('party', [])
            stats['party_size'] = len([p for p in party if p]) if party else 0
            
            # Get badges
            badges = game_data.get('badges', [])
            stats['badges'] = len(badges) if badges else 0
    except:
        pass
    
    # Get milestones
    try:
        response = requests.get('http://localhost:8000/milestones', timeout=1)
        if response.status_code == 200:
            data = response.json()
            # FIXED: Parse new format {'milestones': [{'name': ..., 'completed': True}, ...]}
            if 'milestones' in data and isinstance(data['milestones'], list):
                milestone_list = data['milestones']
                completed = sum(1 for m in milestone_list if m.get('completed', False))
                total = len(milestone_list)
                stats['milestones'] = f"{completed}/{total}"
            else:
                # Fallback for old format
                milestones = data
                completed = sum(1 for v in milestones.values() if isinstance(v, dict) and v.get('completed', False))
                total = len(milestones)
                stats['milestones'] = f"{completed}/{total}"
    except:
        pass
    
    return stats

def main():
    """Live dashboard loop"""
    print("Starting dashboard... (Ctrl+C to exit)")
    time.sleep(2)
    
    try:
        while True:
            clear_screen()
            stats = get_training_stats()
            
            print("=" * 70)
            print(" 🎮 POKEMON EMERALD SPEEDRUN - TRAINING DASHBOARD")
            print("=" * 70)
            print()
            
            # Progress bar with ETA
            pct = stats['progress_pct']
            bar_width = 50
            filled = int(bar_width * pct / 100)
            bar = '█' * filled + '░' * (bar_width - filled)
            print(f"Progress: [{bar}] {pct:.3f}%")
            print(f"Steps:    {stats['current_step']} / {stats['total_steps']}")
            
            # Calculate ETA
            if pct > 0 and 'uptime' in stats:
                steps_done = int(stats['current_step'].replace(',', ''))
                total_steps = 2000000
                steps_per_hour = steps_done / (stats['uptime'] / 3600) if stats['uptime'] > 0 else 0
                if steps_per_hour > 0:
                    remaining_steps = total_steps - steps_done
                    hours_left = remaining_steps / steps_per_hour
                    days = int(hours_left // 24)
                    hours = int(hours_left % 24)
                    print(f"ETA:      ~{days}d {hours}h remaining ({steps_per_hour:.1f} steps/hr)")
            print()
            
            # Core metrics - ENHANCED
            print("📊 TRAINING METRICS")
            print("-" * 70)
            print(f"  Average Reward:         {stats['recent_reward']}")
            print(f"  Milestones Completed:   {stats['milestones']}")
            print(f"  VLM Calls (last 1000):  {stats['vlm_calls']}")
            print(f"  Battle Events:          {stats['battle_events']}")
            print(f"  Episodic Memories:      {stats['episodic_memories']}")
            if 'knn_confidence' in stats:
                print(f"  k-NN Confidence (avg):  {stats['knn_confidence']}")
            if 'stuck_count' in stats:
                print(f"  Stuck Detections:       {stats['stuck_count']}")
            if 'fps' in stats:
                print(f"  Training Speed (FPS):   {stats['fps']}")
            print()
            
            # Agent state - ENHANCED
            print("🤖 AGENT STATE")
            print("-" * 70)
            print(f"  Location:     {stats['map']}")
            print(f"  Position:     {stats['position']}")
            if 'party_size' in stats:
                print(f"  Party Size:   {stats['party_size']}")
            if 'badges' in stats:
                print(f"  Badges:       {stats['badges']}")
            if 'last_action' in stats:
                print(f"  Last Action:  {stats['last_action']}")
            print()
            
            # System health
            print("💚 SYSTEM HEALTH")
            print("-" * 70)
            
            # Check if training process is alive
            import subprocess
            try:
                result = subprocess.run(['pgrep', '-f', 'train_rl_vlm_screen_aware'], 
                                      capture_output=True, text=True)
                if result.stdout.strip():
                    print("  Training Process:   ✓ Running")
                    
                    # Get memory usage
                    pid = result.stdout.strip().split()[0]
                    mem_result = subprocess.run(['ps', '-p', pid, '-o', 'rss='], 
                                              capture_output=True, text=True)
                    if mem_result.stdout.strip():
                        mem_mb = int(mem_result.stdout.strip()) / 1024
                        print(f"  Memory Usage:       {mem_mb:.1f} MB")
                else:
                    print("  Training Process:   ✗ NOT RUNNING")
            except:
                print("  Training Process:   ? Unknown")
            
            # Check server
            try:
                response = requests.get('http://localhost:8000/state', timeout=1)
                if response.status_code == 200:
                    print("  Server API:         ✓ Responding")
                else:
                    print(f"  Server API:         ⚠ Status {response.status_code}")
            except:
                print("  Server API:         ✗ Not responding")
            
            print()
            
            # Recent actions (if available)
            try:
                response = requests.get('http://localhost:8000/recent_actions', timeout=1)
                if response.status_code == 200:
                    actions = response.json()
                    if actions and len(actions) > 0:
                        print("🎮 RECENT ACTIONS (Last 10)")
                        print("-" * 70)
                        recent = actions[-10:] if len(actions) > 10 else actions
                        action_str = ' → '.join([a.get('action', '?') for a in recent])
                        print(f"  {action_str}")
                        print()
            except:
                pass
            
            # Training stability
            print("📈 TRAINING STABILITY")
            print("-" * 70)
            if 'error' in stats:
                print(f"  ⚠️  Error: {stats['error']}")
            else:
                print(f"  Status: ✓ Running smoothly")
            
            # Log file info
            if 'log_file' in stats:
                log_size_mb = os.path.getsize(stats['log_file']) / (1024 * 1024)
                print(f"  Log File: {stats['log_file']}")
                print(f"  Log Size: {log_size_mb:.1f} MB")
            
            if 'uptime' in stats:
                hours = int(stats['uptime'] // 3600)
                minutes = int((stats['uptime'] % 3600) // 60)
                print(f"  Uptime:   {hours}h {minutes}m")
            
            print()
            print("=" * 70)
            print("💡 TIP: Training runs in background with nohup - safe to close terminal!")
            print("=" * 70)
            print("Press Ctrl+C to exit dashboard | Auto-refresh: 5s")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\nDashboard stopped.")

if __name__ == "__main__":
    main()
