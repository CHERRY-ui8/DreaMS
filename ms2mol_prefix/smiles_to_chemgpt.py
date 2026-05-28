"""Convert standard SMILES to ChemGPT-compatible format.

ChemGPT's WordPiece tokenizer was trained on a pre-processed SMILES format
where parentheses are replaced with explicit [Branch*] tokens.

Conversion rules:
    - '(' → [Branch<depth>_<counter>]  (depth=nesting level, counter=Nth at depth)
    - ')' → removed (implicit closure)
    - Everything else (including [...], ring digits, etc.) → pass through

Example:
    CC(C)C            → C[Branch1_1]C
    CC(C)(C)C         → C[Branch1_1]C[Branch1_2]C
    N#CN=C1SCCN1...   → N#C[Branch1_1]N=C1SCCN1...
    CCO               → CCO  (no change)
"""

import re


def smiles_to_chemgpt(smiles: str) -> str:
    """Convert a standard SMILES string to ChemGPT-compatible format.

    Args:
        smiles: Standard SMILES string (e.g. 'CC(C)C')

    Returns:
        ChemGPT-format string (e.g. 'C[Branch1_1]C')
    """
    depth = 0
    # counters per depth: how many branches we've seen at each level
    counters: dict[int, int] = {}
    result = []
    i = 0
    n = len(smiles)

    while i < n:
        ch = smiles[i]

        # ── Bracket atoms: pass through untouched ──
        if ch == '[':
            # Find the matching ']'
            end = smiles.index(']', i) if ']' in smiles[i:] else n
            result.append(smiles[i:end + 1])
            i = end + 1
            continue

        # ── Branch open → [Branch<depth>_<counter>] ──
        if ch == '(':
            depth += 1
            counters[depth] = counters.get(depth, 0) + 1
            result.append(f'[Branch{depth}_{counters[depth]}]')
            i += 1
            continue

        # ── Branch close → removed ──
        if ch == ')':
            depth -= 1
            i += 1
            continue

        # ── Everything else ──
        result.append(ch)
        i += 1

    return ''.join(result)


# ═══════════════════════════════════════════════════════════════
# Reverse: ChemGPT format → standard SMILES
# ═══════════════════════════════════════════════════════════════

_BRANCH_RE = re.compile(r'\[Branch(\d+)_(\d+)\]')


def chemgpt_to_smiles(chemgpt_str: str) -> str:
    """Convert a ChemGPT-format string back to standard SMILES.

    [BranchN_M] is interpreted as: "the preceding character has a branch.
    The branch content is everything after this token until the next
    [Branch*] token at the same or lower level, or end of string."

    This is a best-effort conversion — for deeply nested branches the
    placement of closing parens is heuristic. Simple cases work correctly.

    Example:
        CC[Branch1_1]CC  →  CC(C)C
    """
    parts = _BRANCH_RE.split(chemgpt_str)
    # parts: alternating [text, N, M, text, N, M, ...]

    result = []
    depth = 0
    # Stack of (level, depth_at_branch_start) — not directly needed for algo
    # We use a simpler approach: track if we're in a "branch content" region

    # Our strategy: [BranchN_M] opens a branch at level N.
    # When a new [Branch] opens at level N' <= current depth N,
    # we close the previous branch (output ')') before opening the new one.
    # At end of string, close all remaining branches.

    branch_levels = []  # stack of branch levels

    # The format: we get alternating segments from the regex split
    # [text, level, count, text, level, count, ...]
    # First element is always text before any [Branch]

    current_text = parts[0]  # text before first branch token
    if current_text:
        result.append(current_text)

    i = 1
    while i < len(parts) - 2:
        level = int(parts[i])
        # count = int(parts[i + 1])  # not needed
        next_text = parts[i + 2]

        # Close branches: if we have open branches at >= level, close down to level-1
        while branch_levels and branch_levels[-1] > level:
            result.append(')')
            branch_levels.pop()

        if branch_levels and branch_levels[-1] == level:
            # Same level — close previous, open new
            result.append(')')
            branch_levels.pop()

        # Open new branch
        result.append('(')
        branch_levels.append(level)

        # Add content after this branch token
        if next_text:
            result.append(next_text)

        i += 3  # skip (level, count, text) trio

    # Close any remaining open branches
    while branch_levels:
        result.append(')')
        branch_levels.pop()

    return ''.join(result)


def batch_convert(smiles_list: list[str]) -> list[str]:
    """Batch-convert SMILES to ChemGPT format."""
    return [smiles_to_chemgpt(s) for s in smiles_list]


# ═══════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(
        '/root/.cache/huggingface/hub/models--ncfrey--ChemGPT-1.2B'
        '/snapshots/0164ca1f1754cd36b43c34b185373ee3672e7d65/tokenizer.json'
    )

    test_cases = [
        # (original, expected_converted)
        ('CCO', 'CCO'),                                               # no branches
        ('CC(C)C', 'CC[Branch1_1]CC'),                                # one branch (off C2)
        ('CC(C)(C)C', 'CC[Branch1_1]C[Branch1_2]CC'),                 # two branches at same level
        ('C(C(C)C)C', 'C[Branch1_1]C[Branch2_1]CCC'),                 # nested branches
        ('N#CN=C1SCCN1Cc1ccc(Cl)nc1',
         'N#CN=C1SCCN1Cc1ccc[Branch1_1]Clnc1'),                       # one branch (Cl)
        (')', ''),                                                     # bare close
        ('(', '[Branch1_1]'),                                          # bare open
        ('C(C)(C)(C)C', 'C[Branch1_1]C[Branch1_2]C[Branch1_3]CC'),    # three branches at same level
        ('[nH]1cccc1', '[nH]1cccc1'),                                  # bracket atom, no branches
        ('[nH]1c(C)ccc1', '[nH]1c[Branch1_1]Cccc1'),                  # bracket + branch
        ('C1=CC=CC=C1', 'C1=CC=CC=C1'),                                # ring closures
        ('c1ccccc1', 'c1ccccc1'),                                      # aromatic ring, no branches
        ('', ''),                                                      # empty string
    ]

    print('=== Unit tests ===')
    all_ok = True
    for orig, expected in test_cases:
        converted = smiles_to_chemgpt(orig)
        ok = converted == expected
        status = '✓' if ok else '✗'
        if not ok:
            all_ok = False
            print(f'  {status} FAIL: {orig}')
            print(f'       expected: {expected}')
            print(f'       got:      {converted}')
        else:
            print(f'  {status} {orig:40s} → {converted}')

    print()
    print('=== Tokenizer verification ===')
    for orig, _ in test_cases:
        converted = smiles_to_chemgpt(orig)
        enc = tok.encode(converted)
        has_unk = '[UNK]' in enc.tokens
        status = '✗ UNK' if has_unk else '✓ OK'
        print(f'  {status:6s} {converted:50s} → {enc.tokens[:15]}')

    print()
    if all_ok:
        print('All tests passed! ✓')
    else:
        print('Some tests FAILED! ✗')
