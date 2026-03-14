#!/bin/bash
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Prevent all displays (including DisplayLink) from sleeping
caffeinate -d &

"$CHROME" --app=http://localhost:5001 &
sleep 0.5
# "$CHROME" --app=http://localhost:5002 &
# sleep 0.5
# "$CHROME" --app=http://localhost:5003 &
# sleep 0.5
# "$CHROME" --app=http://localhost:5004 &
