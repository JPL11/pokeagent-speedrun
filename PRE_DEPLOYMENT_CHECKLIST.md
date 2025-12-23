# Pre-Deployment Checklist

## ✅ Completed Items

### Code Implementation
- [x] Strategic directive manager created (283 lines)
- [x] Battle engagement strategy created (353 lines)
- [x] Healing item manager created (323 lines)
- [x] All three modules imported into training file
- [x] Managers initialized in `ServerAwareEnv.__init__()`
- [x] Strategic directive update added at VLM call time (line ~1683)
- [x] VLM prompt enhanced with battle guidance (line ~590-680)
- [x] Context-aware level rewards (8.0 GRIND, 3.0 PROCEED, 5.0 default)
- [x] Context-aware battle win rewards (4.0 GRIND, 1.0 PROCEED)
- [x] Healing item usage rewards (+8.0)
- [x] Strategic goal completion rewards (+50.0)
- [x] Context-aware grass rewards (2.5 GRIND, 0.2 PROCEED)
- [x] Blackout recovery integration
- [x] Syntax validation passed
- [x] Import validation passed

### Documentation
- [x] Implementation summary created (TRAINING_STRATEGY_IMPLEMENTATION.md)
- [x] Pre-deployment checklist created (this file)

## 🔄 When Current Training Completes

### Before Starting New Training:

1. **Verify Current Training Status**
   ```bash
   # Check if 2M training is complete
   tail -50 logs/training_*.log
   ```

2. **Backup Current Code** (Optional)
   ```bash
   cp train_rl_vlm_screen_aware.py train_rl_vlm_screen_aware.py.backup
   ```

3. **Quick Smoke Test** (Optional but recommended)
   ```bash
   # Run 100 steps to verify no runtime errors
   python train_rl_vlm_screen_aware.py --steps 100
   ```
   
   Look for:
   - ✅ Strategic manager initialized
   - ✅ No import errors
   - ✅ Directive updates appear in logs
   - ✅ Context-aware rewards trigger

4. **Start Full Training**
   ```bash
   # Use existing training script
   python train_rl_vlm_screen_aware.py --steps 2000000
   ```

### What to Monitor During Training:

#### Strategic Directives (should appear every ~25 steps)
```
🎯 STRATEGIC DIRECTIVE UPDATED: GRIND
   ↳ Party needs training. Average level 6.0 < target 10. Entering grinding mode.
```

#### VLM Guidance (when VLM is called)
```
🔥 **GRINDING MODE ACTIVE** - Train Pokemon for next challenge!
📊 Current Level: 6.0 | Target: 10
```

#### Context-Aware Rewards
```
⚔️ BATTLE WON! (+4.0 reward during GRINDING)
✨ LEVEL UP! Party gained level (+8.0 reward during GRINDING)
🌿 Walking through grass (+2.5 reward during GRINDING)
```

#### Goal Completion
```
✅ STRATEGIC GOAL COMPLETE! Party reached target level (Lv10)
```

#### Directive Transitions
```
🎯 STRATEGIC DIRECTIVE UPDATED: PROCEED
   ↳ Party is strong enough. Average level 10.2 >= target 10. Resuming story progression.
```

### Expected Behavior:

**Early Game (Levels 1-10)**:
1. Get starter → GRIND directive activates
2. VLM receives grinding guidance
3. Agent seeks tall grass
4. Fights wild Pokemon (rewarded heavily)
5. Levels up to ~10
6. PROCEED directive activates
7. Agent focuses on story progression

**Before Rustboro Gym (Levels 10-13)**:
1. Approaching gym → GRIND directive
2. Agent trains in Route 104 grass
3. Reaches level 13
4. PROCEED directive → enters gym

**Post-Blackout**:
1. Lost battle → POST_BLACKOUT directive
2. Healing emphasis, cautious exploration
3. Gradually returns to GRIND mode

### Performance Metrics to Track:

1. **Average party level over time**
   - Should see steady increase during GRIND
   - Plateaus during PROCEED are OK

2. **Battle win rate**
   - Should improve as party levels up
   - Low win rate → more grinding needed

3. **Milestone completion rate**
   - Should progress through story
   - Gyms should be completable

4. **Blackout frequency**
   - Initial blackouts expected
   - Should decrease as agent learns

## 🔧 Troubleshooting

### Issue: Agent stuck in GRIND mode forever
**Check**: Strategic manager's level threshold
**Fix**: Verify `MILESTONE_LEVEL_REQUIREMENTS` in `strategic_directive_manager.py`

### Issue: Agent still avoiding all battles
**Check**: Rewards appearing in logs?
**Debug**: Add verbose logging in reward section

### Issue: VLM not receiving battle guidance
**Check**: `battle_guidance` variable in prompt
**Debug**: Print `battle_guidance` before VLM call

### Issue: No directive transitions
**Check**: `update_directive()` being called?
**Debug**: Add log at line ~1683 to verify call

## 📊 Success Criteria

After 500K steps, agent should demonstrate:
- ✅ Training Pokemon when under-leveled
- ✅ Progressing story when adequately leveled
- ✅ Using healing items appropriately
- ✅ Recovering from blackouts
- ✅ Completing milestones in order

## 🚀 Deployment Ready

All code changes are implemented and validated. The training system is ready to deploy when current 2M training completes.

**Current Training Status**: 50,600 / 2,000,000 steps (2.53%)

**Estimated Time to Completion**: Several days (depends on hardware)

**Next Action**: Wait for current training to complete, then start new training with strategic battle system.
