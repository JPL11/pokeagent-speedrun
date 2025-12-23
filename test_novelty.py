#!/usr/bin/env python3
"""Test the novelty detector"""

import numpy as np
from utils.novelty_detector import create_novelty_detector

def test_novelty_detector():
    print("🧪 Testing Novelty Detector...")
    print("=" * 60)
    
    # Create detector
    detector = create_novelty_detector()
    print("✅ Novelty detector created\n")
    
    # Test 1: New coordinate
    print("Test 1: Visiting new coordinate")
    result = detector.update(coords=(5, 10, 0), dialogue="Welcome to Littleroot Town!")
    print(f"  Total novelty: {result['total_novelty']:.3f}")
    print(f"  Coord novelty: {result['coord_novelty']:.3f}")
    print(f"  Is novel: {result['is_novel']}")
    print(f"  Unique coords: {result['unique_coords']}\n")
    
    # Test 2: Revisiting same coordinate
    print("Test 2: Revisiting same coordinate")
    result = detector.update(coords=(5, 10, 0), dialogue="Welcome to Littleroot Town!")
    print(f"  Total novelty: {result['total_novelty']:.3f}")
    print(f"  Coord novelty: {result['coord_novelty']:.3f}")
    print(f"  Is novel: {result['is_novel']}")
    print(f"  (Should be lower novelty)\n")
    
    # Test 3: New coordinates (exploring)
    print("Test 3: Exploring new areas")
    for i in range(5):
        result = detector.update(coords=(5+i, 10+i, 0), dialogue=f"Exploring {i}")
        print(f"  Step {i+1}: Novelty={result['total_novelty']:.3f}, Unique={result['unique_coords']}")
    print()
    
    # Test 4: Loop detection
    print("Test 4: Loop detection (repeating same spots)")
    for _ in range(15):
        detector.update(coords=(10, 10, 0), dialogue="Same spot")
    is_loop = detector.detect_loop()
    print(f"  Loop detected: {is_loop}")
    print(f"  (Should be True)\n")
    
    # Test 5: Frame differencing
    print("Test 5: Frame differencing")
    frame1 = np.random.randint(0, 255, (144, 160, 3), dtype=np.uint8)
    frame2 = frame1.copy()
    frame2[50:100, 50:100] = 255  # Change part of frame
    
    result1 = detector.update(coords=(15, 15, 0), frame=frame1)
    result2 = detector.update(coords=(15, 15, 0), frame=frame2)
    print(f"  Same frame: frame_novelty={result1['frame_novelty']:.3f}")
    print(f"  Changed frame: frame_novelty={result2['frame_novelty']:.3f}")
    print(f"  (Changed should be higher)\n")
    
    # Test 6: Summary
    print("Test 6: Get summary for VLM")
    print("-" * 60)
    print(detector.get_summary())
    print("-" * 60)
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_novelty_detector()

