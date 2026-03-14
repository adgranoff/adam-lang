#!/bin/bash
# timed_run.sh — Run a command with elapsed time tracking and optional timeout
#
# Usage: ./tools/timed_run.sh <timeout_minutes> <command...>
#   timeout_minutes: max runtime in minutes (0 = no limit)
#
# Creates .timed_run_status with start time, PID, and command.
# Check elapsed: cat .timed_run_status
# The script updates .timed_run_status with elapsed time every epoch/line of output.

TIMEOUT_MIN=$1
shift
CMD="$@"

STATUS_FILE=".timed_run_status"
START_EPOCH=$(date +%s)
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')

echo "started=$START_EPOCH" > "$STATUS_FILE"
echo "start_time=$START_TIME" >> "$STATUS_FILE"
echo "timeout_min=$TIMEOUT_MIN" >> "$STATUS_FILE"
echo "command=$CMD" >> "$STATUS_FILE"
echo "status=running" >> "$STATUS_FILE"

echo "[timed_run] Started at $START_TIME (timeout: ${TIMEOUT_MIN}m)"
echo "[timed_run] Command: $CMD"

# Run command, pipe output through timestamp updater
if [ "$TIMEOUT_MIN" -gt 0 ]; then
    TIMEOUT_SEC=$((TIMEOUT_MIN * 60))
fi

# Execute command in background
eval "$CMD" &
CMD_PID=$!
echo "pid=$CMD_PID" >> "$STATUS_FILE"

# Monitor loop — check every 30 seconds
while kill -0 $CMD_PID 2>/dev/null; do
    sleep 30
    NOW=$(date +%s)
    ELAPSED=$((NOW - START_EPOCH))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))

    # Update status file with current elapsed
    cat > "$STATUS_FILE" <<EOF
started=$START_EPOCH
start_time=$START_TIME
timeout_min=$TIMEOUT_MIN
command=$CMD
status=running
pid=$CMD_PID
elapsed_sec=$ELAPSED
elapsed=${ELAPSED_MIN}m${ELAPSED_SEC}s
last_check=$(date '+%H:%M:%S')
EOF

    # Check timeout
    if [ "$TIMEOUT_MIN" -gt 0 ] && [ "$ELAPSED" -ge "$TIMEOUT_SEC" ]; then
        echo ""
        echo "[timed_run] TIMEOUT after ${ELAPSED_MIN}m${ELAPSED_SEC}s — killing PID $CMD_PID"
        kill $CMD_PID 2>/dev/null
        sleep 2
        kill -9 $CMD_PID 2>/dev/null
        cat > "$STATUS_FILE" <<EOF
started=$START_EPOCH
start_time=$START_TIME
timeout_min=$TIMEOUT_MIN
command=$CMD
status=killed_timeout
pid=$CMD_PID
elapsed_sec=$ELAPSED
elapsed=${ELAPSED_MIN}m${ELAPSED_SEC}s
ended=$(date '+%Y-%m-%d %H:%M:%S')
EOF
        exit 124
    fi
done

# Command finished naturally
wait $CMD_PID
EXIT_CODE=$?
NOW=$(date +%s)
ELAPSED=$((NOW - START_EPOCH))
ELAPSED_MIN=$((ELAPSED / 60))
ELAPSED_SEC=$((ELAPSED % 60))

cat > "$STATUS_FILE" <<EOF
started=$START_EPOCH
start_time=$START_TIME
timeout_min=$TIMEOUT_MIN
command=$CMD
status=completed
exit_code=$EXIT_CODE
pid=$CMD_PID
elapsed_sec=$ELAPSED
elapsed=${ELAPSED_MIN}m${ELAPSED_SEC}s
ended=$(date '+%Y-%m-%d %H:%M:%S')
EOF

echo ""
echo "[timed_run] Finished in ${ELAPSED_MIN}m${ELAPSED_SEC}s (exit code: $EXIT_CODE)"
exit $EXIT_CODE
