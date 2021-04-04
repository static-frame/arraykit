



import unittest
from enum import Enum


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
    note: genfromtxt only identifies if 'j' is in trailing position, 1+3j, not 3j+1
is string:
    any character other than e, j, n, a, or not true/false sequence
is empty:
    only space
    if there is another float, will be interpreted as NaN
    if there are only integers, will be interpreted as -1
    if combined with False/True, will be interpreted as str
'''

# functions as needed in C implementation
def is_digit(c: str) -> bool:
    return c.isdigit()

def is_alpha(c: str) -> bool:
    return c.isalpha()

def is_space(c: str) -> bool:
    return c.isspace()

def is_sign(c: str) -> bool:
    return c == '+' or c == '-'

def ismatch(c: str, match: str) -> str:
    '''Do a case insensitive character match, given the upper case character.
    '''
    # can go from upper to lower with | 0x20, to upper with & 0x5f
    assert ord(match) <= 90 # must be upper case
    return c == match or c == chr(ord(match) | 0x20)

def is_e(c: str) -> bool:
    return c == 'e' or c == 'E'

def is_j(c: str) -> bool:
    return c == 'j' or c == 'J'

def is_t(c: str) -> bool:
    return c == 't' or c == 'T'

def is_r(c: str) -> bool:
    return c == 'r' or c == 'R'

def is_u(c: str) -> bool:
    return c == 'u' or c == 'U'

def is_f(c: str) -> bool:
    return c == 'f' or c == 'F'

def is_a(c: str) -> bool:
    return c == 'a' or c == 'A'

def is_l(c: str) -> bool:
    return c == 'l' or c == 'L'

def is_s(c: str) -> bool:
    return c == 's' or c == 'S'

class TypeResolved(Enum):
    IS_UNKNOWN = 1
    IS_BOOL = 2
    IS_INT = 3
    IS_FLOAT = 4
    IS_COMPLEX = 5
    IS_STRING = 6
    IS_EMPTY = 7

class TypeField:
    def __init__(self):
        self.resolved: TypeResolved = TypeResolved.IS_UNKNOWN
        self.in_leading_space = False
        self.count_leading_space = 0
        self.count_bool = 0
        self.count_sign = 0

    def process_char(self, c: str, pos: int) -> int:
        # position is postion needs to be  dropping leading space
        # update self based on c and position
        # return int where 1 means process more, 0 means stop, -1 means error

        if self.resolved != TypeResolved.IS_UNKNOWN:
            return 0

        if is_space(c):
            if pos == 0:
                self.in_leading_space = True
                self.count_leading_space += 1
                return 1
            if self.in_leading_space:
                self.count_leading_space += 1
                return 1
        self.in_leading_space = False

        if is_sign(c):
            self.count_sign += 1
            if self.count_sign > 2:
                # complex numbers can have 2 signs, anything else is a string
                self.resolved = TypeResolved.IS_STRING
                return 0
        if is_digit(c):
            return 1

        pos_field = pos - self.count_leading_space

        if pos_field == 0:
            if is_t(c):
                self.count_bool += 1
            if is_f(c):
                self.count_bool -= 1

        if pos_field == 1:
            if is_r(c):
                self.count_bool += 1
            if is_a(c):
                self.count_bool -= 1

        if pos_field == 2:
            if is_u(c):
                self.count_bool += 1
            if is_l(c):
                self.count_bool -= 1

        if pos_field == 3:
            if is_e(c) and self.count_bool == 3:
                self.resolved = TypeResolved.IS_BOOL
                return 0
            if is_s(c):
                self.count_bool -= 1

        if pos_field == 4:
            if is_e(c) and self.count_bool == -4:
                self.resolved = TypeResolved.IS_BOOL
                return 0
        return 1 # keep going

    def process_field(self, field: str) -> TypeResolved:
        for pos, char in enumerate(field):
            if not self.process_char(char, pos):
                break

        if self.resolved == TypeResolved.IS_UNKNOWN:
            return TypeResolved.IS_STRING
        return self.resolved




class TestUnit(unittest.TestCase):

    def test_bool_a(self):
        self.assertEqual(TypeField().process_field('   true'), TypeResolved.IS_BOOL)
        self.assertEqual(TypeField().process_field('FALSE'), TypeResolved.IS_BOOL)
        self.assertEqual(TypeField().process_field('  tals  '), TypeResolved.IS_STRING)


    def test_str_a(self):
        self.assertEqual(TypeField().process_field('+++'), TypeResolved.IS_STRING)



if __name__ == '__main__':
    unittest.main()



