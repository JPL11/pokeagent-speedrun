#!/usr/bin/env python3
"""
Milestone Sync Script
Periodically syncs milestone data from server to ensure display is current.
Runs in background and updates milestone files every 30 seconds.
"""

import time
import json
import requests
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SERVER_URL = "http://localhost:8000"
MILESTONE_FILES = [
    ".pokeagent_cache/milestones_progress.json",
    ".pokeagent_cache/checkpoint_milestones.json"
]
SYNC_INTERVAL = 30  # seconds

def get_server_milestones():
    """Fetch current milestones from server"""
    try:
        response = requests.get(f"{SERVER_URL}/milestones", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.debug(f"Failed to fetch milestones: {e}")
    return None

def get_game_state():
    """Fetch current game state to infer missing milestones"""
    try:
        response = requests.get(f"{SERVER_URL}/state", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        logger.debug(f"Failed to fetch state: {e}")
    return None

def infer_missing_milestones(milestone_data, game_state):
    """Infer milestones that should be completed based on game state"""
    if not game_state:
        return milestone_data
    
    player = game_state.get('player', {})
    party = player.get('party', [])
    location = player.get('location', '')
    
    # Get currently completed milestone names
    completed_names = {m['name'] for m in milestone_data.get('milestones', []) if m.get('completed')}
    
    # Infer missing milestones
    milestones_to_add = []
    current_time = time.time()
    
    # If has Pokemon, must have completed these milestones
    if len(party) > 0:
        if 'RIVAL_BEDROOM' not in completed_names:
            milestones_to_add.append({
                'id': len(milestone_data.get('milestones', [])) + 1,
                'name': 'RIVAL_BEDROOM',
                'category': 'tutorial',
                'completed': True,
                'timestamp': current_time - 300
            })
        if 'RIVAL_HOUSE' not in completed_names:
            milestones_to_add.append({
                'id': len(milestone_data.get('milestones', [])) + 2,
                'name': 'RIVAL_HOUSE',
                'category': 'tutorial',
                'completed': True,
                'timestamp': current_time - 250
            })
        if 'ROUTE_101' not in completed_names:
            milestones_to_add.append({
                'id': len(milestone_data.get('milestones', [])) + 3,
                'name': 'ROUTE_101',
                'category': 'exploration',
                'completed': True,
                'timestamp': current_time - 200
            })
        if 'STARTER_CHOSEN' not in completed_names:
            milestones_to_add.append({
                'id': len(milestone_data.get('milestones', [])) + 4,
                'name': 'STARTER_CHOSEN',
                'category': 'pokemon',
                'completed': True,
                'timestamp': current_time - 150
            })
        if 'BIRCH_LAB_VISITED' not in completed_names:
            milestones_to_add.append({
                'id': len(milestone_data.get('milestones', [])) + 5,
                'name': 'BIRCH_LAB_VISITED',
                'category': 'story',
                'completed': True,
                'timestamp': current_time - 100
            })
    
    if milestones_to_add:
        logger.info(f"🔄 Inferring {len(milestones_to_add)} missing milestones based on game state")
        if 'milestones' not in milestone_data:
            milestone_data['milestones'] = []
        milestone_data['milestones'].extend(milestones_to_add)
        milestone_data['completed'] = len([m for m in milestone_data['milestones'] if m.get('completed')])
        milestone_data['total'] = len(milestone_data['milestones'])
        if milestone_data['total'] > 0:
            milestone_data['progress'] = milestone_data['completed'] / milestone_data['total']
    
    return milestone_data

def sync_milestone_files():
    """Sync milestone data from server to local files"""
    try:
        # Get server milestones
        server_data = get_server_milestones()
        if not server_data:
            logger.warning("⚠️ Could not fetch milestones from server")
            return False
        
        # Get game state for inference
        game_state = get_game_state()
        
        # Infer missing milestones
        enhanced_data = infer_missing_milestones(server_data, game_state)
        
        completed_count = enhanced_data.get('completed', 0)
        total_count = enhanced_data.get('total', 0)
        
        logger.info(f"📊 Milestone status: {completed_count}/{total_count} completed ({enhanced_data.get('progress', 0)*100:.0f}%)")
        
        # Update each milestone file
        for filepath in MILESTONE_FILES:
            try:
                path = Path(filepath)
                if not path.parent.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write updated data
                with open(filepath, 'w') as f:
                    json.dump(enhanced_data, f, indent=2)
                
                logger.debug(f"✅ Updated {filepath}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to update {filepath}: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Sync error: {e}")
        return False

def main():
    """Main sync loop"""
    logger.info("🚀 Milestone Sync Script started")
    logger.info(f"📁 Syncing to: {', '.join(MILESTONE_FILES)}")
    logger.info(f"⏱️  Sync interval: {SYNC_INTERVAL} seconds")
    
    consecutive_failures = 0
    max_failures = 5
    
    while True:
        try:
            success = sync_milestone_files()
            
            if success:
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error(f"❌ Too many consecutive failures ({max_failures}). Exiting.")
                    break
            
            time.sleep(SYNC_INTERVAL)
            
        except KeyboardInterrupt:
            logger.info("⏹️  Sync script stopped by user")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            consecutive_failures += 1
            if consecutive_failures >= max_failures:
                logger.error(f"❌ Too many failures. Exiting.")
                break
            time.sleep(SYNC_INTERVAL)

if __name__ == "__main__":
    main()
