#!/bin/bash
trap 'echo "Kill everyone!"; kill 0' EXIT
python3 Simulator/simulator.py > /dev/null 2>&1 &
python3 Reciever/reciever.py > /dev/null 2>&1 &
python3 Business/business.py > /dev/null 2>&1 &
python3 GUI/dash_app_test.py
