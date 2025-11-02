"""
Verify that the phoneme class configuration is correct for 44-class recognizer.

Expected structure:
- 39 ARPAbet phonemes (aa, ae, ah, ao, aw, ay, b, ch, d, dh, eh, er, ey, f, g, hh, ih, iy, jh, k, l, m, n, ng, ow, oy, p, r, s, sh, t, th, uh, uw, v, w, y, z, zh)
- 1 silence token (sil)
- 1 error token (err)
- 3 special tokens (<blank>, <bos>, <eos>)
Total: 39 + 1 + 1 + 3 = 44 classes
"""

def load_arpa_phonemes(path="data/arpa_phonemes"):
    """Load phonemes from arpa_phonemes file."""
    phonemes = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if parts:
                    phonemes.append(parts[0])
    return phonemes


def main():
    print("=" * 60)
    print("PHONEME CLASS VERIFICATION")
    print("=" * 60)
    
    # Load ARPAbet phonemes
    arpa_phonemes = load_arpa_phonemes()
    print(f"\n1. ARPAbet phonemes from data/arpa_phonemes: {len(arpa_phonemes)} phonemes")
    print(f"   Phonemes: {', '.join(arpa_phonemes)}")
    
    # Expected phonemes
    expected_base_phonemes = [
        'aa', 'ae', 'ah', 'ao', 'aw', 'ay',  # vowels
        'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh',  # consonants
        'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy',
        'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh'
    ]
    
    print(f"\n2. Expected ARPAbet phonemes: {len(expected_base_phonemes)} phonemes")
    
    # Check for sil
    has_sil = 'sil' in arpa_phonemes
    print(f"\n3. Silence token (sil): {'✓ Found' if has_sil else '✗ Missing'}")
    
    # Check for err
    has_err = 'err' in arpa_phonemes
    print(f"4. Error token (err): {'✓ Found' if has_err else '✗ Missing'}")
    
    # Special tokens
    special_tokens = ['<blank>', '<bos>', '<eos>']
    print(f"\n5. Special tokens (added by label encoder): {len(special_tokens)} tokens")
    print(f"   Tokens: {', '.join(special_tokens)}")
    
    # Calculate total
    total_classes = len(arpa_phonemes) + len(special_tokens)
    print(f"\n{'=' * 60}")
    print(f"TOTAL CLASSES: {total_classes}")
    print(f"{'=' * 60}")
    
    if total_classes == 44:
        print("✓ Correct! 44-class phoneme recognizer")
    else:
        print(f"✗ Error! Expected 44 classes, got {total_classes}")
    
    # Breakdown
    print(f"\nBreakdown:")
    print(f"  - ARPAbet phonemes: {len(expected_base_phonemes)}")
    print(f"  - Silence (sil): 1")
    print(f"  - Error (err): 1")
    print(f"  - Special tokens (<blank>, <bos>, <eos>): 3")
    print(f"  - Total: {len(expected_base_phonemes)} + 1 + 1 + 3 = {len(expected_base_phonemes) + 1 + 1 + 3}")
    
    # Check for missing or extra phonemes
    arpa_set = set(arpa_phonemes)
    expected_set = set(expected_base_phonemes + ['sil', 'err'])
    
    missing = expected_set - arpa_set
    extra = arpa_set - expected_set
    
    if missing:
        print(f"\n⚠ Missing phonemes: {', '.join(sorted(missing))}")
    if extra:
        print(f"\n⚠ Extra phonemes: {', '.join(sorted(extra))}")
    
    if not missing and not extra:
        print("\n✓ All expected phonemes are present, no extra phonemes found!")


if __name__ == "__main__":
    main()
