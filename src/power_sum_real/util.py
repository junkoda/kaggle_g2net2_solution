from typing import Tuple


def get_range(ra: str, n: int) -> Tuple[int, int]:
    """
    istart, iend = get_range('1:10', n)

    Args:
      ra (str or None): 1, 1:, 5:10
      n (int): default iend when it is omitted, ra = '1:'
    """

    if ra is None:
        return 0, n
    elif ra.isnumeric():
        # Single number
        i = int(ra)
        return i, i + 1
    elif ':' in ra:
        v = ra.split(':')
        assert len(v) == 2
        istart = 0 if v[0] == '' else int(v[0])
        iend = n if v[1] == '' else int(v[1])
        return istart, iend
    else:
        raise ValueError('Failed to parse range: {}'.format(ra))
