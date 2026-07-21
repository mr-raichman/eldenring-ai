"""
watch_lockon_candidates.py

Watches all scanmem candidate addresses simultaneously.
Toggle lock-on in game - only the real one will flip.

Usage:
    python tools/diagnose_camera.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eldenring_ai.io.memory import find_pid, _read_bytes

FIELD_OFFSET = 0x111  # confirmed from AOB scan

# Paste scanmem addresses here (the first column)
CANDIDATES = [
    0x55557ae0b909,
    0x55557ae0b939,
    0x55557ae0b969,
    0x7ffe96a73e10,
    0x7ffe96a760e4,
    0x7ffe9726eae0,
    0x7ffe9726eb48,
    0x7ffe9726ebb0,
    0x7ffe97667cb0,
    0x7ffe97667d50,
    0x7ffe97667ed0,
    0x7ffe97668700,
    0x7ffe976687a0,
    0x7ffe97669300,
    0x7ffe97669480,
    0x7ffe97669520,
    0x7ffea03d7820,
    0x7ffea04075f6,
    0x7fffb1b9c6b0,
    0x7fffb1b9c6b1,
    0x7fffb2f22c8b,
    0x7fffb548f729,
    0x7fffb613edb0,
]

def read_byte(pid, addr):
    try:
        return _read_bytes(pid, addr, 1)[0]
    except OSError:
        return None


def main():
    pid = find_pid()
    if pid is None:
        print("Elden Ring not found.")
        return
    print(f"PID: {pid}")
    print(f"Watching {len(CANDIDATES)} candidates. Toggle lock-on and watch for changes.")
    print("Press Ctrl+C to stop.\n")

    prev = {addr: read_byte(pid, addr) for addr in CANDIDATES}

    print("Initial values:")
    for addr in CANDIDATES:
        print(f"  0x{addr:X} = {prev[addr]}")
    print()

    input("Press Enter to begin watching...")
    print()

    confirmed = []

    try:
        while True:
            time.sleep(0.05)
            for addr in CANDIDATES:
                val = read_byte(pid, addr)
                if val != prev[addr]:
                    ts = time.strftime("%H:%M:%S")
                    struct_base = addr - FIELD_OFFSET
                    print(f"  [{ts}]  0x{addr:X}  {prev[addr]} → {val}  "
                          f"(struct base = 0x{struct_base:X})")
                    if addr not in confirmed and val not in (None, 0):
                        confirmed.append(addr)
                    prev[addr] = val
    except KeyboardInterrupt:
        pass

    print("\n─── Summary ───")
    if confirmed:
        print("Addresses that went nonzero (lock-on ON candidates):")
        for addr in confirmed:
            struct_base = addr - FIELD_OFFSET
            print(f"  Lock byte:   0x{addr:X}")
            print(f"  Struct base: 0x{struct_base:X}")
        print("\nNext step: run diagnose_pointer_chain.py with these struct base addresses.")
    else:
        print("No changes detected. Try toggling lock-on while this is running.")


if __name__ == "__main__":
    main()
