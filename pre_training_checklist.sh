#!/bin/bash
# Pre-Training Checklist and Startup
# Run this before starting enhanced training

echo "========================================================================"
echo "PRE-TRAINING CHECKLIST"
echo "========================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

checks_passed=0
checks_total=0

# Check 1: Server running
echo -n "1. Checking if server is running... "
checks_total=$((checks_total + 1))
if ps aux | grep "server.app" | grep -v grep > /dev/null; then
    echo -e "${GREEN}✓ Running${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${RED}✗ Not running${NC}"
    echo "   Start with: source .venv/bin/activate && python -m server.app --port 8000 &"
fi

# Check 2: Code compiles
echo -n "2. Checking if code compiles... "
checks_total=$((checks_total + 1))
if python -c "import py_compile; py_compile.compile('train_rl_vlm_screen_aware.py', doraise=True)" 2>/dev/null; then
    echo -e "${GREEN}✓ Compiles${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${RED}✗ Syntax error${NC}"
fi

# Check 3: EpisodicMemory loads
echo -n "3. Checking EpisodicMemory loads... "
checks_total=$((checks_total + 1))
if python -c "from agent.episodic_memory import EpisodicMemory; EpisodicMemory()" 2>/dev/null; then
    echo -e "${GREEN}✓ Loads${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${RED}✗ Import error${NC}"
fi

# Check 4: Virtual environment active
echo -n "4. Checking virtual environment... "
checks_total=$((checks_total + 1))
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${GREEN}✓ Active${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${YELLOW}⚠ Not active${NC}"
    echo "   Run: source .venv/bin/activate"
fi

# Check 5: Disk space
echo -n "5. Checking disk space... "
checks_total=$((checks_total + 1))
available_gb=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [[ $available_gb -gt 20 ]]; then
    echo -e "${GREEN}✓ ${available_gb}GB available${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${YELLOW}⚠ Only ${available_gb}GB available${NC}"
fi

# Check 6: No training already running
echo -n "6. Checking for existing training... "
checks_total=$((checks_total + 1))
if ps aux | grep "train_rl" | grep -v grep > /dev/null; then
    echo -e "${RED}✗ Training already running!${NC}"
    echo "   Kill with: pkill -f train_rl"
else
    echo -e "${GREEN}✓ No conflicts${NC}"
    checks_passed=$((checks_passed + 1))
fi

# Check 7: CUDA available (optional)
echo -n "7. Checking CUDA availability... "
checks_total=$((checks_total + 1))
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo -e "${GREEN}✓ CUDA available${NC}"
    checks_passed=$((checks_passed + 1))
else
    echo -e "${YELLOW}⚠ CPU only (slower)${NC}"
    checks_passed=$((checks_passed + 1))  # Don't fail on this
fi

echo ""
echo "========================================================================"
echo "Checklist: $checks_passed/$checks_total passed"
echo "========================================================================"
echo ""

if [[ $checks_passed -ge 6 ]]; then
    echo -e "${GREEN}✓ Ready to start training!${NC}"
    echo ""
    echo "Start with:"
    echo "  ./start_enhanced_training.sh            # Interactive"
    echo "  nohup ./start_enhanced_training.sh &    # Background"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Please fix issues above before starting training${NC}"
    echo ""
    exit 1
fi
