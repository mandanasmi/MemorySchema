#!/bin/bash
# Monitor ConSpec training progress

echo "======================================"
echo "ConSpec Training Monitor"
echo "======================================"
echo ""
echo "Wandb Dashboard: https://wandb.ai/mandanasmi/schema-learning"
echo ""
echo "Training Log (last 50 lines):"
echo "--------------------------------------"
tail -50 training_output.log
echo ""
echo "======================================"
echo "To view live updates, run:"
echo "  tail -f training_output.log"
echo "======================================"

