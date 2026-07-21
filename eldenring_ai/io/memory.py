"""
memory.py - reading game state without an API. Finds the Elden Ring process,
AOB-scans for the WorldChrMan static pointer, walks the pointer chain to the
player struct for HP / stamina / area id, and reads the boss HP bar from the
captured frame with OpenCV.
"""

import os
import struct

import cv2
import numpy as np

from eldenring_ai.config import paths
from eldenring_ai.config.vision import BOSS_HP_REGION, BOSS_HP_CAP_FULL
from eldenring_ai.config.offsets import (
    _WORLD_CHR_MAN_AOB,
    MARGIT_AREA_ID,
    WCM_TO_PTR1,
    PTR1_TO_PTR2,
    PTR2_TO_PTR3,
    PTR3_TO_PTR4,
    PTR4_CURRENT_HP,
    PTR4_MAX_HP,
    PTR4_IS_READY,
    PTR4_AREA_ID,
    PTR4_STAMINA_CUR,
    PTR4_STAMINA_MAX,
)

_boss_hp_history = []

def reset_boss_hp_smoothing():
    global _boss_hp_history
    _boss_hp_history = []

def _read_boss_hp_bar(frame):
    global _boss_hp_history
    r    = BOSS_HP_REGION
    bar  = frame[r["y1"]:r["y2"], r["x1"]:r["x2"]]
    grey = cv2.cvtColor(bar, cv2.COLOR_BGR2GRAY)

    BRIGHTNESS_THRESHOLD = 120
    SATURATION_THRESHOLD = 60

    hsv          = cv2.cvtColor(bar, cv2.COLOR_BGR2HSV)
    col_max_grey = grey.max(axis=0)
    col_max_sat  = hsv[:, :, 1].max(axis=0)

    valid_cols = np.where(
        (col_max_grey > BRIGHTNESS_THRESHOLD) &
        (col_max_sat  < SATURATION_THRESHOLD)
    )[0]

    if len(valid_cols) == 0:
        raw = _boss_hp_history[-1] if _boss_hp_history else 1.0
    else:
        cap_x = valid_cols[0]
        raw   = min(1.0, round(cap_x / BOSS_HP_CAP_FULL, 3))
    _boss_hp_history.append(raw)
    if len(_boss_hp_history) > 5:
        _boss_hp_history.pop(0)

    return float(np.median(_boss_hp_history))

def _read_bytes(pid, address, size):
    """Open /proc/<pid>/mem and read `size` bytes starting at `address`."""
    with open(f"/proc/{pid}/mem", "rb") as mem:
        mem.seek(address)
        return mem.read(size)


def _read_int32(pid, address):
    """Read a 4-byte signed integer (little-endian). Used for HP values."""
    return struct.unpack("<i", _read_bytes(pid, address, 4))[0]


def _read_int64(pid, address):
    """Read an 8-byte unsigned integer (little-endian). Used for pointers."""
    return struct.unpack("<Q", _read_bytes(pid, address, 8))[0]

def find_pid(process_name="eldenring.exe"):
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="replace")
            if process_name.lower() in cmdline.lower():
                return int(entry)
        except (PermissionError, FileNotFoundError):
            continue
    return None

def _parse_maps(pid):
    regions = []
    with open(f"/proc/{pid}/maps") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            start, end = [int(x, 16) for x in parts[0].split("-")]
            perms = parts[1]
            path  = " ".join(parts[5:]) if len(parts) > 5 else ""
            regions.append({
                "start": start,
                "end":   end,
                "size":  end - start,
                "perms": perms,
                "path":  path,
            })
    return regions


def find_base_address(pid, module_name="eldenring.exe"):
    regions   = _parse_maps(pid)
    candidates = [
        r["start"] for r in regions
        if module_name.lower() in r["path"].lower()
    ]
    return min(candidates) if candidates else None


def _get_scan_regions(pid):
    regions = _parse_maps(pid)
    base    = find_base_address(pid)
    if base is None:
        return []

    lower_bound  = max(0, base - 0x100000000)
    upper_bound  = base + 0x100000000
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


def _aob_scan(pid, pattern_str, regions):
    tokens  = pattern_str.strip().split()
    pattern = []
    for token in tokens:
        if token == "??":
            pattern.append((0, True))
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


def find_world_chr_man(pid):
    """
    AOB-scan for the instruction that loads WorldChrMan, then extract
    the RIP-relative pointer address using the embedded 32-bit offset.

    The matched instruction is: MOV RAX, [RIP + rel32]
    Encoding:  48 8B 05  [XX XX XX XX]
    The 4-byte signed offset at +3 is relative to the instruction end (+7),
    so:   ptr_address = match_addr + 7 + relative_offset

    Returns the address of the static pointer variable, or None if
    the pattern is not found or the pointer currently holds zero.
    """
    regions    = _get_scan_regions(pid)

    match_addr = _aob_scan(pid, _WORLD_CHR_MAN_AOB, regions)
    if match_addr is None:
        return None

    relative_offset = struct.unpack("<i", _read_bytes(pid, match_addr + 3, 4))[0]
    ptr_address     = match_addr + 7 + relative_offset

    # Sanity check: the pointer should currently hold a non-zero address.
    world_chr_man = _read_int64(pid, ptr_address)

    return ptr_address if world_chr_man != 0 else None

CACHE_FILE = str(paths.PTR_CACHE)

def find_world_chr_man_cached(pid):
    base = find_base_address(pid)

    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE) as f:
            cached_base, cached_offset = f.read().split()
            cached_base   = int(cached_base, 16)
            cached_offset = int(cached_offset, 16)

        if cached_base == base:
            ptr_address   = base + cached_offset
            world_chr_man = _read_int64(pid, ptr_address)
            if world_chr_man != 0:
                return ptr_address

    ptr_address = find_world_chr_man(pid)
    if ptr_address is not None:
        offset = ptr_address - base
        with open(CACHE_FILE, "w") as f:
            f.write(f"{base:X} {offset:X}")

        return ptr_address

    return None

class GameMemory:
    def __init__(self):
        self.pid                    = None
        self._world_chr_man_ptr     = None

        self._last_ptr4             = None
        self._last_current_hp       = None
        self._last_max_hp           = None
        self._last_is_ready         = 0
        self._last_area_id          = 0

    def connect(self):
        self.pid = find_pid()
        if self.pid is None:
            return False

        self._world_chr_man_ptr = find_world_chr_man_cached(self.pid)

        if self._world_chr_man_ptr is None:
            return False
        return True

    def resolve_static_ptr(self):
        if self.pid is None:
            return False
        regions = _get_scan_regions(self.pid)
        match_addr = _aob_scan(self.pid, _WORLD_CHR_MAN_AOB, regions)
        if match_addr is None:
            return False
        rel = struct.unpack("<i", _read_bytes(self.pid, match_addr + 3, 4))[0]
        self._world_chr_man_ptr = match_addr + 7 + rel
        return True

    def world_loaded(self):
        if self.pid is None or self._world_chr_man_ptr is None:
            return False
        try:
            return _read_int64(self.pid, self._world_chr_man_ptr) != 0
        except OSError:
            return False

    def refresh(self):
        if self.pid is None or self._world_chr_man_ptr is None:
            self._clear_player_state()
            return

        try:
            world_chr_man = _read_int64(self.pid, self._world_chr_man_ptr)
            if world_chr_man == 0:
                self._clear_player_state()
                return

            ptr1 = _read_int64(self.pid, world_chr_man + WCM_TO_PTR1)
            if ptr1 == 0:
                self._clear_player_state()
                return

            ptr2 = _read_int64(self.pid, ptr1 + PTR1_TO_PTR2)
            if ptr2 == 0:
                self._clear_player_state()
                return

            ptr3 = _read_int64(self.pid, ptr2 + PTR2_TO_PTR3)
            if ptr3 == 0:
                self._clear_player_state()
                return

            ptr4 = _read_int64(self.pid, ptr3 + PTR3_TO_PTR4)
            if ptr4 == 0:
                self._clear_player_state()
                return

            self._last_ptr4       = ptr4
            self._last_current_hp = _read_int32(self.pid, ptr4 + PTR4_CURRENT_HP)
            self._last_max_hp     = _read_int32(self.pid, ptr4 + PTR4_MAX_HP)
            self._last_is_ready   = _read_int32(self.pid, ptr4 + PTR4_IS_READY)
            self._last_area_id    = _read_bytes(self.pid, ptr4 + PTR4_AREA_ID, 1)[0]

        except OSError:
            self._clear_player_state()

    def _clear_player_state(self):
        """Reset all cached player state to the loading-screen safe defaults."""
        self._last_ptr4       = None
        self._last_current_hp = None
        self._last_max_hp     = None
        self._last_is_ready   = 0
        self._last_area_id    = 0

    def is_player_ready(self):
        return (
            self._last_ptr4 is not None
            and self._last_current_hp is not None
            and self._last_current_hp > 0
            and self._last_max_hp is not None
            and self._last_max_hp > 0
            and self._last_current_hp >= self._last_max_hp
            and self._last_is_ready == 1
            and self._last_area_id != 0
        )

    def loading_screen(self):
        return (
            self._last_area_id == 0
        )

    def is_in_boss_arena(self):
        return (
            self._last_ptr4 is not None
            and self._last_area_id == MARGIT_AREA_ID
        )

    def read_player_hp(self):
        if self._last_current_hp is None or not self._last_max_hp:
            return 1.0
        ratio = self._last_current_hp / self._last_max_hp
        return round(max(0.0, min(1.0, ratio)), 3)

    def read_stamina(self):
        if self._last_ptr4 is None or self._last_current_hp is None:
            return 1.0
        try:
            current = _read_int32(self.pid, self._last_ptr4 + PTR4_STAMINA_CUR)
            maximum = _read_int32(self.pid, self._last_ptr4 + PTR4_STAMINA_MAX)
            if maximum <= 0:
                return 1.0
            return round(max(0.0, min(1.0, current / maximum)), 3)
        except OSError:
            return 1.0

def get_game_state(frame, memory: GameMemory):
    return (
        memory.read_player_hp(),
        _read_boss_hp_bar(frame),
        memory.read_stamina(),
    )
