from typing import Sequence

def keep_all(arr):
    return list(arr)


def keep_lower(arr):
    num = (len(arr) + 1) // 2
    return list(sorted(arr))[:num]


def keep_latter_half(arr):
    num = (len(arr) + 1) // 2
    return list(arr)[-num:]


def seqeq(seq1: Sequence, seq2: Sequence):
    for a, b in zip(seq1, seq2):
        if a != b:
            return False
    return True


def iter_subset(s: int, include_emtpy_set=False):
    ss = s
    while ss != 0:
        yield ss
        ss = (ss - 1) & s
    if include_emtpy_set:
        yield 0

