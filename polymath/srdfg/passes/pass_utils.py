import numba

def is_string(var):
    if var[0] == '"' and var[-1] == '"':
        return True
    else:
        return False

def is_number(var):
    try:
        float(var)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(var)
        return True
    except (TypeError, ValueError):
        pass
    return False


def is_literal(var):
    str  = is_string(var)
    num = is_number(var)
    if str or num:
        return True
    else:
        return False