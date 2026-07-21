"""
find_lockon_chain.py

For each lock-on struct base candidate, scans memory for pointers to it,
then checks if those pointers are themselves reachable from a static
(game-module) address. Reports complete pointer chains.

Usage:
    python tools/diagnose_camera2.py
"""

import os
import sys
import struct
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eldenring_ai.io.memory import find_pid, find_base_address, _parse_maps, _read_bytes, _read_int64

FIELD_OFFSET = 0x111

# All struct bases derived from watch output (address - 0x111)
STRUCT_BASES = [
    0x7FFE9726E9CF,
    0x7FFE9726EA37,
    0x7FFE9726EA9F,
    0x7FFE97667B9F,
    0x7FFE97667C3F,
    0x7FFE97667DBF,
    0x7FFE9766936F,
    0x7FFE9766940F,
    0x7FFEA03D770F,
    0x7FFFB1B9C59F,
    0x7FFFB2F22B7A,
    0x7FFFB548F618,
    0x7FFEA04074E5,
    0x7FFE96A73CFF,
    0x7FFE96A75FD3,
    0x7FFE976685EF,
    0x7FFE9766868F,
    0x7FFE976691EF,
    0x7FFFB613EC9F,
    0x7FFFB1B9C5A0,
]

POINTER_TOLERANCE = 0x200   # a pointer within ±this of struct base counts
MAX_CHAIN_DEPTH   = 3       # how many levels deep to chase


def get_all_readable_regions(pid):
    regions = _parse_maps(pid)
    result  = []
    for r in regions:
        if "r" not in r["perms"]:
            continue
        if r["path"] in ("[vvar]", "[vsyscall]"):
            continue
        if r["end"] - r["start"] > 0x20000000:  # skip >512MB
            continue
        result.append((r["start"], r["end"], r["perms"], r["path"]))
    return result


def get_module_regions(pid, base):
    """Return only regions that are part of the game module (static memory)."""
    regions = _parse_maps(pid)
    result  = []
    for r in regions:
        if "eldenring.exe" in r["path"].lower():
            result.append((r["start"], r["end"]))
    return result


def scan_for_pointers_to(pid, target_addr, tolerance, regions):
    """
    Scan all regions for 8-byte values within tolerance of target_addr.
    Returns list of addresses that contain such a pointer.
    """
    lo = target_addr - tolerance
    hi = target_addr + tolerance
    hits = []

    for (start, end, perms, path) in regions:
        size = end - start
        try:
            data = _read_bytes(pid, start, size)
        except OSError:
            continue

        # Walk 8 bytes at a time (aligned)
        for i in range(0, len(data) - 7, 8):
            val = struct.unpack_from("<Q", data, i)[0]
            if lo <= val <= hi:
                hits.append((start + i, val))

    return hits


def is_in_module(addr, module_regions):
    return any(start <= addr < end for start, end in module_regions)


def main():
    print("=" * 60)
    print("  Lock-On Pointer Chain Finder")
    print("=" * 60 + "\n")

    pid = find_pid()
    if pid is None:
        print("Elden Ring not found.")
        return
    print(f"PID: {pid}")

    base = find_base_address(pid)
    print(f"Base: 0x{base:X}\n")

    print("Loading memory regions...")
    all_regions    = get_all_readable_regions(pid)
    module_regions = get_module_regions(pid, base)
    total_mb = sum(e - s for s, e, *_ in all_regions) // (1024 * 1024)
    print(f"  {len(all_regions)} regions, {total_mb} MB total\n")

    print("Scanning for pointer chains. This will take 1-2 minutes...\n")

    results = []

    for struct_base in STRUCT_BASES:
        lock_byte = struct_base + FIELD_OFFSET
        print(f"Trying struct base 0x{struct_base:X} (lock byte @ 0x{lock_byte:X})...")

        # Level 1: find pointers directly to struct base
        level1 = scan_for_pointers_to(pid, struct_base, POINTER_TOLERANCE, all_regions)
        if not level1:
            print(f"  No level-1 pointers found.\n")
            continue
        print(f"  Level-1 hits: {len(level1)}")

        # Check if any level-1 pointer is in the game module (static → struct)
        static_direct = [(addr, val) for addr, val in level1 if is_in_module(addr, module_regions)]
        if static_direct:
            for addr, val in static_direct:
                offset = val - struct_base
                print(f"  ✓ STATIC DIRECT: base+0x{addr-base:X} → 0x{val:X} (+0x{offset:X})")
                results.append({
                    "struct_base": struct_base,
                    "chain": [("STATIC", base, addr - base, val, offset)],
                })

        # Level 2: for each level-1 hit, scan for pointers to it
        for ptr1_addr, ptr1_val in level1[:50]:  # limit to avoid excessive scan time
            level2 = scan_for_pointers_to(pid, ptr1_addr, 0, all_regions)
            static_l2 = [(addr, val) for addr, val in level2 if is_in_module(addr, module_regions)]
            for addr, val in static_l2:
                offset1 = ptr1_val - struct_base
                print(f"  ✓ STATIC 2-LEVEL:")
                print(f"      base+0x{addr-base:X} → 0x{ptr1_addr:X} → 0x{ptr1_val:X} (+0x{offset1:X}) → lock byte")
                results.append({
                    "struct_base": struct_base,
                    "chain": [
                        ("STATIC", base, addr - base, ptr1_addr, 0),
                        ("OFFSET", ptr1_addr, 0, ptr1_val, offset1),
                    ],
                })

        print()

    print("\n" + "=" * 60)
    if results:
        print(f"Found {len(results)} chain(s):\n")
        for r in results:
            sb = r["struct_base"]
            print(f"  Struct base: 0x{sb:X}")
            print(f"  Lock byte:   0x{sb + FIELD_OFFSET:X}")
            for step in r["chain"]:
                print(f"    {step}")
            print()
        print("Add the confirmed chain to memory.py and read:")
        print("  _read_bytes(pid, resolved_struct_base + 0x111, 1)[0]")
    else:
        print("No static pointer chains found.")
        print("The structure is likely allocated dynamically with no static root.")
        print("Consider using scanmem's pointer scanner instead:")
        print(f"  sudo scanmem -p {pid}")
        print(f"  > pointer <one of the lock byte addresses>")


if __name__ == "__main__":
    main()
