"""
offsets.py - game memory constants: WorldChrMan AOB signature and the
pointer-chain offsets used to read player state out of the game process.

Single source of truth for the numbers that io/memory.py walks. The x86
instruction-decoding internals (the RIP-relative +3/+7 math and the AOB scan
window) stay in memory.py - they are not tunable, they describe the encoding.
"""

# AOB signature for the instruction that loads WorldChrMan.
_WORLD_CHR_MAN_AOB = "48 8B 05 ?? ?? ?? ?? 48 85 C0 74 0F 48 39 88"

# Pointer chain: WorldChrMan -> ptr1 -> ptr2 -> ptr3 -> ptr4 (player struct).
WCM_TO_PTR1 = 0x10EF8
PTR1_TO_PTR2 = 0x000
PTR2_TO_PTR3 = 0x190
PTR3_TO_PTR4 = 0x000

# Fields read off ptr4 (the player struct).
PTR4_CURRENT_HP = 0x138
PTR4_MAX_HP = 0x13C
PTR4_IS_READY = 0x0A0
PTR4_AREA_ID = 0x19A
PTR4_STAMINA_CUR = 0x154
PTR4_STAMINA_MAX = 0x158

# Area id that identifies the Margit arena.
MARGIT_AREA_ID = 4

# Area id at the pre-Margit grace. After a death the respawn transiently reports the
# arena id while loading out of the death location, so the reset workflow waits for
# this settled value before walking to the fog (else it can start an episode outside).
BEFORE_MARGIT_AREA_ID = 68
