from typing import Sequence


def seqeq(seq1: Sequence, seq2: Sequence):
    """
    Compare whether two sequences are equal
    """
    for a, b in zip(seq1, seq2):
        if a != b:
            return False
    return True


def iter_subset(s: int, include_emtpy_set=False):
    """
    Iterate the subset of a set represented by the binary representation of s.
    """
    ss = s
    while ss != 0:
        yield ss
        ss = (ss - 1) & s
    if include_emtpy_set:
        yield 0

