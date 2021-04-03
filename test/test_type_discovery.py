






'''
Discovery

options:
is bool: all values are case insensitive "true" or "false"
is int:
    all values are numbers
    might start with "+" or "-"
    might permit comma
is float:
    all values are numbers except
    might have one "e" or "E" between numbers (cannot start or end)
    might have one ".", can lead or trail
    can have "nan"
is complex:
    all values are numbers except
    must have one "j" or "J", cannot lead, can trail; if used as delimiter, must be followed by sign
    might have one or two "+", "-", leading or after j
    might be surrounded in parenthesis
is string:
    any character other than e, j, n, a, or not true/false sequence
'''










if __name__ == '__main__':
    pass