#!/usr/bin/env python3
"""
Quick sanity check script - validates the agent architecture is working correctly
"""

import requests
import time
import json

def check_architecture():
    """Verify all components are active"""
    print("🔍 Architecture Sanity Check\n")
    
    checks = {
        'Server API': False,
        'VLM Oracle': False,
        'SNN Active': False,
        'k-NN Memory': False,
        'Battle Tracker': False,
        'Milestone Progress': False
    }
    
    # 1. Check server is responding
    try:
        response = requests.get('http://localhost:8000/state', timeout=2)
        if response.status_code == 200:
            checks['Server API'] = True
            state = response.json()
            location = state.get('player', {}).get('location', 'UNKNOWN')
            print(f"✓ Server API: {location}")
        else:
            print(f"✗ Server API: Status {response.status_code}")
    except Exception as e:
        print(f"✗ Server API: {e}")
    
    # 2. Check VLM has been called
    try:
        response = requests.get('http://localhost:8000/agent', timeout=2)
        if response.status_code == 200:
            agent_data = response.json()
            recent = agent_data.get('recent_interactions', [])
            if recent and len(recent) > 0:
                checks['VLM Oracle'] = True
                last_action = recent[0].get('response', '')[:50]
                print(f"✓ VLM Oracle: Called (latest: {last_action}...)")
            else:
                print("✗ VLM Oracle: No interactions yet")
        else:
            print(f"✗ VLM Oracle: Status {response.status_code}")
    except Exception as e:
        print(f"✗ VLM Oracle: {e}")
    
    # 3. Check training log for SNN activity
    try:
        with open('training_2M_steps.log', 'r') as f:
            lines = f.readlines()[-500:]  # Last 500 lines
            
            # Check for SNN mentions
            snn_mentions = [l for l in lines if 'SNN' in l and 'neuron' in l]
            if snn_mentions:
                checks['SNN Active'] = True
                print(f"✓ SNN Active: {len(snn_mentions)} log entries")
            else:
                print("✗ SNN Active: No mentions in recent logs")
            
            # Check for k-NN/episodic memory
            knn_mentions = [l for l in lines if 'Episodic' in l or 'k-NN' in l]
            if knn_mentions:
                checks['k-NN Memory'] = True
                print(f"✓ k-NN Memory: {len(knn_mentions)} log entries")
            else:
                print("✗ k-NN Memory: No mentions in recent logs")
            
            # Check for battle events
            battle_mentions = [l for l in lines if 'battle' in l.lower() and ('won' in l or 'lost' in l or 'event' in l)]
            if battle_mentions:
                checks['Battle Tracker'] = True
                print(f"✓ Battle Tracker: {len(battle_mentions)} events")
            else:
                print("⚠ Battle Tracker: No battle events yet (may be early)")
    except FileNotFoundError:
        print("✗ Training log not found - training may not have started")
    except Exception as e:
        print(f"✗ Log check failed: {e}")
    
    # 4. Check milestones
    try:
        response = requests.get('http://localhost:8000/milestones', timeout=2)
        if response.status_code == 200:
            milestones = response.json()
            completed = sum(1 for v in milestones.values() if isinstance(v, bool) and v)
            total = sum(1 for v in milestones.values() if isinstance(v, bool))
            checks['Milestone Progress'] = True
            print(f"✓ Milestone Progress: {completed}/{total} completed")
        else:
            print(f"✗ Milestone Progress: Status {response.status_code}")
    except Exception as e:
        print(f"✗ Milestone Progress: {e}")
    
    # Summary
    print("\n" + "="*50)
    passed = sum(checks.values())
    total = len(checks)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("✅ All systems operational!")
        return True
    elif passed >= 4:
        print("⚠️ Most systems working, some may be early in training")
        return True
    else:
        print("❌ Multiple systems not responding - check logs")
        return False

def check_progress():
    """Check if agent is making forward progress"""
    print("\n🎮 Progress Check\n")
    
    try:
        with open('training_2M_steps.log', 'r') as f:
            lines = f.readlines()[-1000:]
            
            # Extract steps
            import re
            steps = [int(m.group(1)) for l in lines for m in [re.search(r'Step (\d+)', l)] if m]
            if steps and len(steps) > 2:
                print(f"✓ Training active: Steps {min(steps)} → {max(steps)}")
                
                # Check step rate
                if len(steps) > 10:
                    step_diff = max(steps) - min(steps)
                    print(f"  Progress rate: ~{step_diff} steps in last {len(lines)} log lines")
            
            # Check coordinates
            coords = [m.group(1) for l in lines for m in [re.search(r'Calling VLM oracle at (\([0-9]+,[0-9]+\))', l)] if m]
            if coords:
                unique_coords = len(set(coords))
                print(f"✓ Exploration: {unique_coords} unique positions visited recently")
                print(f"  Latest positions: {coords[-5:]}")
            
            # Check rewards
            rewards = [float(m.group(1)) for l in lines for m in [re.search(r'reward[:\s]+([0-9.-]+)', l, re.I)] if m]
            if rewards:
                avg_reward = sum(rewards) / len(rewards)
                print(f"✓ Rewards: avg={avg_reward:.2f}, count={len(rewards)}")
            
            # Check for stuck patterns
            if len(coords) > 10:
                last_10 = coords[-10:]
                if len(set(last_10)) <= 3:
                    print(f"⚠️ Possible stuck: Only {len(set(last_10))} unique positions in last 10 VLM calls")
            
    except Exception as e:
        print(f"✗ Progress check failed: {e}")

if __name__ == "__main__":
    print("="*50)
    print("ARCHITECTURE SANITY CHECK")
    print("="*50 + "\n")
    
    all_good = check_architecture()
    check_progress()
    
    print("\n" + "="*50)
    if all_good:
        print("✅ System ready for long-term training")
        print("Monitor with: tail -f training_2M_steps.log")
        print("Or watch: cat monitor.log")
    else:
        print("⚠️ Some issues detected - review above")
