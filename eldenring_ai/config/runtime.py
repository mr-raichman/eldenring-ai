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
ARENA_CONFIRM_SECONDS = 1.0
OUTSIDE_ARENA_LIMIT = 5

# Recovery and readiness timings, in wall-clock seconds. The *_TIMEOUT values bound
# how long reset() waits for a game state before giving up and retrying; the
# *_INTERVAL values are how often it re-reads memory while waiting.
CONTROLLER_SETTLE_DELAY    = 2.0   # after creating the virtual pad, before first use
CONDITION_POLL_INTERVAL    = 0.1   # _wait_for_condition re-read cadence
GAME_LAUNCH_POLL_INTERVAL  = 3.0   # while waiting for the game PID to appear
GAME_KILL_POLL_INTERVAL    = 0.2   # while waiting for a killed game process to disappear
PTR_RESOLVE_POLL_INTERVAL  = 1.0   # while re-resolving the static pointer
PLAYER_READY_POLL_INTERVAL = 0.2   # while waiting for the player to become controllable
ARENA_CONFIRM_INTERVAL     = 0.1   # area_id re-read cadence inside _confirm_in_arena

LOADING_SCREEN_TIMEOUT = 10.0
PLAYER_READY_TIMEOUT   = 15.0
BEFORE_MARGIT_TIMEOUT  = 20.0
ARENA_ENTER_TIMEOUT    = 2.0
POST_RECOVERY_DELAY    = 1.0       # settle pause after the launch/recovery branch
CAPTURE_RETRY_DELAY    = 1.0       # after tearing down a failed ScreenCapture

# Sanity floor for the save backup before it is copied over the live save. The real
# file is a fixed-size ~29 MB container; anything far below that is a bad copy, and
# restoring it would break the character rather than reset it.
SAVE_MIN_BYTES = 20 * 1024 * 1024

# Scripted-sequence timings, hand-measured against real game pacing. These are
# calibration values, not derived constants - re-measure if the route, the game
# speed, or the character's stamina changes.
#
# walk_to_fog: pre-Margit grace to the fog gate.
FOG_TURN1_DURATION       = 0.3
FOG_TURN1_SPEED          = 30000
FOG_SPRINT_LEG1_DURATION = 2.0
FOG_TURN2_DURATION       = 0.5
FOG_TURN2_SPEED          = 20000
FOG_SPRINT_LEG2_DURATION = 1.75
FOG_GATE_LOAD_DELAY      = 3.5     # fog-gate transition load

# use_grace_return_item: pouch menu navigation beats.
POUCH_OPEN_DELAY      = 0.5
POUCH_HIGHLIGHT_DELAY = 0.1
POUCH_RELEASE_DELAY   = 0.25
POUCH_CONFIRM_DELAY   = 0.25
