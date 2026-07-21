import os
import struct
import cv2
import numpy as np


# ── Boss HP via OpenCV ─────────────────────────────────────────────────────────
# We read boss HP as a visual ratio because finding Margit's HP in memory
# proved unreliable. Boss bar visibility is now determined by memory (area ID),
# so OpenCV is only used for the HP ratio value - never for visibility.

BOSS_HP_REGION = {
    "x1": 464, "y1": 870,   # 6-pixel tall region for noise resistance
    "x2": 1463, "y2": 876,
}


def _read_boss_hp_bar(frame):
    """
    Read the boss HP ratio from the screen using OpenCV.
    Returns a float 0.0-1.0. This is used only for the reward delta;
    visibility is determined separately via memory (area ID byte).
    """
    r = BOSS_HP_REGION
    bar = frame[r["y1"]:r["y2"], r["x1"]:r["x2"]]
    hsv = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    lower_black = np.array([0,   0,   0])
    upper_black = np.array([180, 255, 50])
    mask = cv2.inRange(hsv, lower_black, upper_black)
    black_pixels = cv2.countNonZero(mask)
    total_pixels = bar.shape[1] * bar.shape[0]
    empty_ratio = black_pixels / total_pixels
    return round(1.0 - empty_ratio, 3)


# ── Low-level memory reading ───────────────────────────────────────────────────

def _read_bytes(pid, address, size):
    """Open the process memory file and read `size` bytes at `address`."""
    with open(f"/proc/{pid}/mem", "rb") as mem:
        mem.seek(address)
        return mem.read(size)


def _read_int32(pid, address):
    """Read a 4-byte signed integer (little-endian). Used for HP values."""
    return struct.unpack("<i", _read_bytes(pid, address, 4))[0]


def _read_int64(pid, address):
    """Read an 8-byte unsigned integer (little-endian). Used for pointers."""
    return struct.unpack("<Q", _read_bytes(pid, address, 8))[0]


# ── Process and address discovery ─────────────────────────────────────────────

def find_pid(process_name="eldenring.exe"):
    """
    Scan /proc for a process whose cmdline contains process_name.
    Returns the PID as an integer, or None if not found.
    """
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                # Arguments are null-separated; replace for easy string search
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
            if process_name.lower() in cmdline.lower():
                return int(entry)
        except (PermissionError, FileNotFoundError):
            continue
    return None


def _parse_maps(pid):
    """
    Parse /proc/<pid>/maps into a list of region dicts.
    Each dict has: start, end, size, perms, path.
    Paths with spaces (e.g. 'ELDEN RING/Game') are preserved correctly
    by joining all tokens from index 5 onward.
    """
    regions = []
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            start, end = [int(x, 16) for x in parts[0].split("-")]
            perms = parts[1]
            path = " ".join(parts[5:]) if len(parts) > 5 else ""
            regions.append({
                "start": start,
                "end":   end,
                "size":  end - start,
                "perms": perms,
                "path":  path,
            })
    return regions


def find_base_address(pid, module_name="eldenring.exe"):
    """
    Under Wine/Proton, eldenring.exe is mapped as small labelled PE-header
    regions. The lowest such region's start address is the module base.
    """
    regions = _parse_maps(pid)
    candidates = [
        r["start"] for r in regions
        if module_name.lower() in r["path"].lower()
    ]
    return min(candidates) if candidates else None


def _get_scan_regions(pid):
    """
    Return (start, size) tuples for memory regions worth scanning for AOBs.
    Under Wine/Proton the game's executable code lives in anonymous rwxp
    regions near the eldenring.exe base address, not in named file mappings.
    We include both the small PE-header regions and all anonymous executable
    regions within 4GB of the base.
    """
    regions = _parse_maps(pid)
    base = find_base_address(pid)
    if base is None:
        return []

    lower_bound = max(0, base - 0x100000000)
    upper_bound = base + 0x100000000

    scan_regions = []
    for r in regions:
        if "r" not in r["perms"]:
            continue
        in_range     = lower_bound <= r["start"] <= upper_bound
        is_pe_header = "eldenring.exe" in r["path"].lower()
        is_anon_exec = (r["path"] == "" and "x" in r["perms"])
        if is_pe_header or (is_anon_exec and in_range):
            scan_regions.append((r["start"], r["size"]))

    return scan_regions


# ── AOB scanning ──────────────────────────────────────────────────────────────

def _aob_scan(pid, pattern_str, regions):
    """
    Scan `regions` for a byte pattern where '??' is a wildcard.
    Returns the absolute address of the first match, or None.
    """
    tokens = pattern_str.strip().split()
    pattern = []
    for token in tokens:
        if token == "??":
            pattern.append((0, True))   # wildcard
        else:
            pattern.append((int(token, 16), False))
    pat_len = len(pattern)

    for (start, size) in regions:
        try:
            data = _read_bytes(pid, start, size)
        except OSError:
            continue
        for i in range(len(data) - pat_len):
            if all(
                is_wc or data[i + j] == byte
                for j, (byte, is_wc) in enumerate(pattern)
            ):
                return start + i

    return None


# ── WorldChrMan discovery ─────────────────────────────────────────────────────
# WorldChrMan is the game's main character manager singleton. All character
# objects (player, enemies) are reachable from it.
# We store the *static pointer address* (not the value it points to) so that
# if WorldChrMan is rebuilt after a loading screen, refresh() always reads the
# current address rather than a stale cached one.

_WORLD_CHR_MAN_AOB = "48 8B 05 ?? ?? ?? ?? 48 85 C0 74 0F 48 39 88"


def find_world_chr_man(pid):
    """
    AOB-scan for the instruction that loads WorldChrMan, extract the
    RIP-relative pointer address, and return it.
    Returns the address of the static pointer (not the WorldChrMan value),
    or None if the pattern is not found.
    """
    regions = _get_scan_regions(pid)
    print(f"  Scanning {len(regions)} regions for WorldChrMan AOB...")

    match_addr = _aob_scan(pid, _WORLD_CHR_MAN_AOB, regions)
    if match_addr is None:
        return None

    print(f"  AOB match at: 0x{match_addr:X}")

    # The matched instruction is: MOV RAX, [RIP + rel32]
    # Bytes: 48 8B 05 [XX XX XX XX]
    # The 4-byte signed offset at +3 is relative to the instruction's end (+7).
    relative_offset = struct.unpack("<i", _read_bytes(pid, match_addr + 3, 4))[0]
    ptr_address = match_addr + 7 + relative_offset
    print(f"  Static pointer address: 0x{ptr_address:X}")

    # Sanity check: verify the pointer currently holds a valid address
    world_chr_man = _read_int64(pid, ptr_address)
    print(f"  WorldChrMan currently at: 0x{world_chr_man:X}")

    return ptr_address if world_chr_man != 0 else None


# ── Public interface ──────────────────────────────────────────────────────────

class GameMemory:
    """
    Manages all memory reads from the Elden Ring process.

    Pointer chain to the player object (PlayerIns / ptr4):
      WorldChrMan + 0x10EF8  →  ptr1  (dereference)
      ptr1        + 0x000    →  ptr2  (dereference at offset 0)
      ptr2        + 0x190    →  ptr3  (dereference)
      ptr3        + 0x000    →  ptr4  (PlayerIns, dereference at offset 0)

    From ptr4, all player state is read at fixed offsets confirmed on v1.16.1:
      + 0x138  current HP (int32)
      + 0x13C  max HP     (int32)
      + 0x0A0  ready flag (int32): 0 = loading, 1 = fully controllable
      + 0x19A  area ID    (byte):  0 = loading/transition,
                                   4 = Margit's arena,
                                  68 = approach corridor outside fog gate

    Flask data is read via a separate static pointer (hardcoded for v1.16.1):
      flask_object + 0x388  flask count (byte, floors at 1 even when empty)
      flask_object + 0x3B0  flask empty flag (byte): 1 = has flasks, 0 = empty
    """

    def __init__(self):
        self.pid                   = None
        self._world_chr_man_ptr    = None  # static pointer address, never changes

        # Cached values populated by refresh() each step
        self._last_ptr4            = None
        self._last_current_hp      = None
        self._last_max_hp          = None
        self._last_is_ready        = 0
        self._last_area_id         = 0

        # Flask tracking via separate static pointer (v1.16.1 hardcoded)
        self._flask_ptr            = 0x6FFFFA12B880
        self._last_flasks          = 0
        self._last_flask_empty_flag = 1   # safe default: assume flasks available

    def connect(self):
        """
        Find the Elden Ring process and locate WorldChrMan.
        Must be called once before any refresh() calls.
        Returns True on success, False if the game isn't running.
        """
        self.pid = find_pid()
        if self.pid is None:
            print("Elden Ring process not found.")
            return False
        print(f"Found Elden Ring: PID {self.pid}")

        self._world_chr_man_ptr = find_world_chr_man(self.pid)
        if self._world_chr_man_ptr is None:
            print("WorldChrMan not found - is the game fully in-world?")
            return False

        print(f"Connected. WorldChrMan pointer at: 0x{self._world_chr_man_ptr:X}")
        return True

    def refresh(self):
        """
        Walk the pointer chain and update all cached values.
        Call this once per step before reading any state.
        If any pointer in the chain is null (loading screen),
        _last_ptr4 is set to None and all derived values are cleared.
        Flask data is read independently so a flask read failure
        never disrupts the main character chain.
        """
        if self.pid is None or self._world_chr_man_ptr is None:
            self._last_ptr4         = None
            self._last_current_hp   = None
            self._last_max_hp       = None
            self._last_is_ready     = 0
            self._last_area_id      = 0
            return

        try:
            # Re-read WorldChrMan from the static pointer every call.
            # This means if WorldChrMan moves after a loading screen
            # we always follow the current address.
            world_chr_man = _read_int64(self.pid, self._world_chr_man_ptr)
            if world_chr_man == 0:
                self._last_ptr4 = None
                return

            ptr1 = _read_int64(self.pid, world_chr_man + 0x10EF8)
            if ptr1 == 0:
                self._last_ptr4 = None
                return

            ptr2 = _read_int64(self.pid, ptr1 + 0)
            if ptr2 == 0:
                self._last_ptr4 = None
                return

            ptr3 = _read_int64(self.pid, ptr2 + 0x190)
            if ptr3 == 0:
                self._last_ptr4 = None
                return

            ptr4 = _read_int64(self.pid, ptr3 + 0)
            if ptr4 == 0:
                self._last_ptr4 = None
                return

            # All pointers resolved - read player state
            self._last_ptr4       = ptr4
            self._last_current_hp = _read_int32(self.pid, ptr4 + 0x138)
            self._last_max_hp     = _read_int32(self.pid, ptr4 + 0x13C)
            self._last_is_ready   = _read_int32(self.pid, ptr4 + 0x0A0)
            self._last_area_id    = _read_bytes(self.pid, ptr4 + 0x19A, 1)[0]

        except OSError:
            # Game may have closed or a region became unreadable
            self._last_ptr4       = None
            self._last_current_hp = None
            self._last_max_hp     = None
            self._last_is_ready   = 0
            self._last_area_id    = 0

        # Flask data - separate try/except so failure here doesn't affect
        # the main character state reads above
        try:
            flask_obj = _read_int64(self.pid, self._flask_ptr)
            if flask_obj:
                self._last_flasks           = _read_bytes(self.pid, flask_obj + 0x388, 1)[0]
                self._last_flask_empty_flag = _read_bytes(self.pid, flask_obj + 0x3B0, 1)[0]
        except OSError:
            pass

    # ── State query methods ────────────────────────────────────────────────────
    # All methods read cached values set by refresh(). Call refresh() first.

    def is_player_loaded(self):
        """
        True if the player character object exists in memory.
        False during loading screens and area transitions (ptr4 is null).
        """
        return self._last_ptr4 is not None

    def is_player_alive(self):
        """True if the player object exists and current HP is above zero."""
        return (
            self._last_ptr4 is not None and
            self._last_current_hp is not None and
            self._last_current_hp > 0
        )

    def is_player_at_full_hp(self):
        """
        True if the player is alive and at maximum HP.
        Useful for detecting a fresh respawn at a grace site.
        """
        return (
            self._last_ptr4 is not None and
            self._last_current_hp is not None and
            self._last_max_hp is not None and
            self._last_max_hp > 0 and
            self._last_current_hp >= self._last_max_hp
        )

    def is_player_ready(self):
        """
        True when the player is fully in control and in a stable area.

        Requires two independent signals to both be satisfied:
          1. Ready flag (ptr4 + 0x0A0 == 1): game has finished loading
             the player and handed control back. Discovered by comparing
             memory snapshots at 'just spawned/frozen' vs 'can move'.
          2. Area ID non-zero (ptr4 + 0x19A != 0): player is in a
             stable loaded area, not in a transition. During respawn
             the area ID briefly reads 4 (arena) before settling to
             the grace corridor - requiring non-zero here filters that out,
             but the full is_in_boss_arena() check handles the arena case.
        """
        return (
            self._last_ptr4 is not None and
            self._last_is_ready == 1 and
            self._last_area_id != 0
        )

    def is_in_boss_arena(self):
        """
        True when the player is inside Margit's boss arena.

        ptr4 + 0x19A is an area ID byte confirmed on v1.16.1:
          4  = inside Margit's arena (fog gate crossed, fight active)
          68 = approach corridor just outside the fog gate
          0  = loading screen or area transition
        The brief flicker where area_id reads 4 during respawn (before
        settling to 68) is filtered by the consecutive-reads requirement
        in _verify_in_arena(), not here.
        """
        return (
            self._last_ptr4 is not None and
            self._last_area_id == 4
        )

    def read_player_hp(self):
        """
        Returns player HP as a float ratio 0.0-1.0.
        Returns 1.0 as a safe fallback when memory can't be read,
        preventing false death detections during loading.
        """
        if not self.is_player_loaded():
            return 1.0
        if not self._last_max_hp:
            return 1.0
        ratio = self._last_current_hp / self._last_max_hp
        return round(max(0.0, min(1.0, ratio)), 3)

    def has_flasks(self):
        """
        True if the player has at least one Crimson Tear flask remaining.
        Uses the binary empty flag at flask_object + 0x3B0.
        This flag goes cleanly 1→0 when the last flask is consumed,
        while the count byte floors at 1 and is therefore unreliable for
        detecting the truly-empty state.
        """
        return self._last_flask_empty_flag == 1


def get_game_state(frame, memory: GameMemory):
    """
    Build the game state dict for one step.

    IMPORTANT: does NOT call memory.refresh() internally.
    The caller must call memory.refresh() before calling this function
    so that the cached values are current.

    boss_bar_visible uses memory (is_in_boss_arena) rather than OpenCV
    to eliminate false positives from dark screens or UI changes.
    boss_hp is still read from OpenCV since Margit's HP in memory
    was not reliably located.
    """
    boss_hp = _read_boss_hp_bar(frame)
    return {
        "player_hp":        memory.read_player_hp(),
        "boss_hp":          boss_hp,
        "boss_bar_visible": memory.is_in_boss_arena(),  # memory-based
        "player_loaded":    memory.is_player_loaded(),
        "player_alive":     memory.is_player_alive(),
        "player_full_hp":   memory.is_player_at_full_hp(),
    }
