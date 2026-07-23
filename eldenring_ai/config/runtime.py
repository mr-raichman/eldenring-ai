"""
runtime.py - runtime behavior toggles and game-launch parameters.
"""

DEBUG_MODE = 0

# Rolling number of most-recent episodes kept in the per-step record CSV.
STEP_RECORD_EPISODES = 5

ACTION_LOCK_DURATION = 0.2

ELDEN_RING_APP_ID = "1245620"
GAME_LAUNCH_TIMEOUT = 180  # seconds
MENU_CONFIRM_INTERVAL = 1.0  # seconds between BTN_A presses

# Arena safety against a start on a transient area_id read: how long area_id must
# stay in the arena before starting an episode, and how many consecutive out-of-arena
# steps abort a running one (self-heal - reset() then retries).
ARENA_CONFIRM_SECONDS = 3.0
OUTSIDE_ARENA_LIMIT = 5
