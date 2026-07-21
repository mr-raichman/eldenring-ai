"""
find_flask_ptr.py - Walk backwards from known flask count addresses
to find a stable static pointer chain rooted in a base module address.

Paste the addresses found by scanmem into CANDIDATE_ADDRESSES below,
then run with: sudo python find_flask_ptr.py
"""

import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eldenring_ai.io.memory import find_pid, find_base_address, _parse_maps, _read_bytes, _read_int64

# ── Paste your scanmem addresses here ─────────────────────────────────────────

CANDIDATE_ADDRESSES = [
    0x7fff683c7728,
    0x7fff9ecad6d4,
    0x7fff9fd16890,
]

# How far back from each candidate to search for pointers pointing to it.
# A pointer to address X will be somewhere in the range [X - MAX_OFFSET, X].
MAX_OFFSET = 0x800

# ── Helpers ────────────────────────────────────────────────────────────────────

def read_int64(pid, address):
    try:
        return struct.unpack("<Q", _read_bytes(pid, address, 8))[0]
    except OSError:
        return None


def find_pointers_to(pid, target, regions, max_offset=MAX_OFFSET):
    """
    Scan all readable regions for 8-byte values that point into
    [target - max_offset, target]. Returns list of (ptr_address, offset) tuples.
    """
    results = []
    low  = target - max_offset
    high = target

    for (start, size) in regions:
        try:
            data = _read_bytes(pid, start, size)
        except OSError:
            continue
        # Walk through the region in 8-byte steps
        for i in range(0, len(data) - 8, 8):
            value = struct.unpack_from("<Q", data, i)[0]
            if low <= value <= high:
                offset = target - value
                results.append((start + i, offset))

    return results


def is_static(address, base, maps):
    """
    Return True if `address` falls inside a named module region
    (PE header or mapped file) - these are stable across restarts.
    """
    for r in maps:
        if r["start"] <= address < r["end"]:
            if r["path"] and "eldenring" in r["path"].lower():
                return True
            # Also accept addresses very close to the module base
            if abs(address - base) < 0x10000000:
                return True
    return False


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    pid = find_pid()
    if pid is None:
        print("Elden Ring not found. Make sure the game is running.")
        sys.exit(1)
    print(f"Found Elden Ring: PID {pid}\n")

    base = find_base_address(pid)
    print(f"Module base address: 0x{base:X}\n")

    maps    = _parse_maps(pid)
    regions = [
        (r["start"], r["size"])
        for r in maps
        if "r" in r["perms"] and r["size"] < 0x10000000
    ]
    print(f"Scanning {len(regions)} readable regions...\n")

    for candidate in CANDIDATE_ADDRESSES:
        print(f"{'═' * 60}")
        print(f"Candidate: 0x{candidate:X}")
        print(f"{'─' * 60}")

        # Level 1: find what points to the candidate
        level1 = find_pointers_to(pid, candidate, regions)
        if not level1:
            print("  No pointers found pointing to this address.\n")
            continue

        print(f"  Found {len(level1)} pointer(s) at level 1:")
        for ptr_addr, offset in level1[:10]:   # show max 10
            static = is_static(ptr_addr, base, maps)
            tag    = " ← STATIC (promising!)" if static else ""
            print(f"    0x{ptr_addr:X}  (offset +0x{offset:X}){tag}")

        # Level 2: find what points to each level-1 pointer
        print(f"\n  Searching level 2...")
        for ptr_addr, offset1 in level1[:5]:
            level2 = find_pointers_to(pid, ptr_addr, regions)
            for ptr2_addr, offset2 in level2[:5]:
                static = is_static(ptr2_addr, base, maps)
                tag    = " ← STATIC (promising!)" if static else ""
                print(f"    0x{ptr2_addr:X}  →  0x{ptr_addr:X}+0x{offset2:X}  →  candidate+0x{offset1:X}{tag}")

        print()


if __name__ == "__main__":
    main()
