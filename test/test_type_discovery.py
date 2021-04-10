



import unittest
import typing as tp
from enum import Enum


'''
Discovery

options:
is bool:
    all values are case insensitive "true" or "false"
    genfromtxt, dtype None: requires no leading space, ignores trailing space
    genfromtxt, dtype bool: requires four character true with no leading space
    AK: can permit leading spaces
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

'''
Discover contiguous numeric, i.e., if contiguous sequence of digits, e, j, sign,decimal; then, after parse complete, look at e/j/decimal counts to determine numeric type
'''


# functions as needed in C implementation
def is_digit(c: str) -> bool:
    #define isdigit_ascii(c) (((unsigned)(c) - '0') < 10u)
    return c.isdigit()

def is_space(c: str) -> bool:
    #define isspace_ascii(c) (((c) == ' ') || (((unsigned)(c) - '\t') < 5))
    return c.isspace()

def is_sign(c: str) -> bool:
    return c == '+' or c == '-'

def is_decimal(c: str) -> bool:
    return c == '.'

# def ismatch(c: str, match: str) -> str:
#     '''Do a case insensitive character match, given the upper case character.
#     '''
#     # can go from upper to lower with | 0x20, to upper with & 0x5f
#     assert ord(match) <= 90 # must be upper case
#     return c == match or c == chr(ord(match) | 0x20)

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
    '''
    Estimate the type of a field. This estimate can be based on character type counts. Some ordering considerations will be ignored for convenience; if downstream parsing fails, fallback will be to a string type anyway.
    '''
    def __init__(self) -> None:
        self.reset()
        self.resolved_line: TypeResolved = TypeResolved.IS_UNKNOWN

    def reset(self) -> None:
        self.resolved_field: TypeResolved = TypeResolved.IS_UNKNOWN

        self.previous_leading_space = False
        self.previous_numeric = False
        self.contiguous_numeric = False

        self.count_leading_space = 0
        self.count_bool = 0 # signed, not greater than +/- 5

        # numeric symbols; values do not need to be greater than 2
        self.count_sign = 0
        self.count_e = 0
        self.count_j = 0
        self.count_decimal = 0

    def process_char(self, c: str, pos: int) -> int:
        # position is postion needs to be  dropping leading space
        # update self based on c and position
        # return int where 1 means process more, 0 means stop, -1 means error

        if self.resolved_field != TypeResolved.IS_UNKNOWN:
            return 0

        # evaluate space -------------------------------------------------------
        space = False

        if is_space(c):
            if pos == 0:
                self.previous_leading_space = True
                self.count_leading_space += 1
                return 1
            if self.previous_leading_space:
                self.count_leading_space += 1
                return 1
            space = True

        self.previous_leading_space = False # this char is not space

        pos_field = pos - self.count_leading_space

        # evaluate numeric -----------------------------------------------------
        numeric = False

        if is_digit(c):
            numeric = True

        elif is_sign(c):
            self.count_sign += 1
            if self.count_sign > 2:
                # complex numbers can have 2 signs, anything else is a string
                self.resolved_field = TypeResolved.IS_STRING
                return 0
            numeric = True

        elif is_e(c): # only character that is numeric and bool
            numeric = True
            self.count_e += 1
            if pos_field == 0 or self.count_e > 1:
                # true and false each only have one E
                self.resolved_field = TypeResolved.IS_STRING
                return 0

        elif is_j(c):
            numeric = True
            self.count_j += 1
            if pos_field == 0 or self.count_j > 1:
                self.resolved_field = TypeResolved.IS_STRING
                return 0

        elif is_decimal(c):
            numeric = True
            self.count_decimal += 1
            if self.count_decimal > 2: # complex can have 2!
                self.resolved_field = TypeResolved.IS_STRING
                return 0

        #-----------------------------------------------------------------------
        print(f' pre: {c=} {pos=} {pos_field=} {numeric=} {self.previous_numeric=} {self.contiguous_numeric=}')
        if numeric:
            if pos_field == 0:
                self.contiguous_numeric = True
                self.previous_numeric = True
                return 1 # E can not be in first position
            # pos_field > 0
            if not self.previous_numeric:
                # found a numeric not in pos 0 where previous was not numeric
                self.contiguous_numeric = False

            self.previous_numeric = True
            # NOTE: we need to consider possible Boolean scenario
            if self.contiguous_numeric or not is_e(c):
                return 1
        else:
            if self.contiguous_numeric and not space:
                # if we find a non-numeric, non-space, after contiguous numeric
                self.resolved_field = TypeResolved.IS_STRING
                return 0
            self.previous_numeric = False

        print(f'post: {c=} {pos=} {pos_field=} {numeric=} {self.previous_numeric=} {self.contiguous_numeric=}')

        # evaluate character positions -----------------------------------------

        if pos_field == 0:
            if is_t(c):
                self.count_bool += 1
            if is_f(c):
                self.count_bool -= 1

        elif pos_field == 1:
            if is_r(c):
                self.count_bool += 1
            if is_a(c):
                self.count_bool -= 1

        elif pos_field == 2:
            if is_u(c):
                self.count_bool += 1
            if is_l(c):
                self.count_bool -= 1

        elif pos_field == 3:
            if is_e(c) and self.count_bool == 3:
                self.resolved_field = TypeResolved.IS_BOOL
                return 0
            if is_s(c):
                self.count_bool -= 1

        elif pos_field == 4:
            if is_e(c) and self.count_bool == -4:
                self.resolved_field = TypeResolved.IS_BOOL
                return 0

        return 1 # keep going

    def resolve_field_type(self, count: int) -> None:
        '''
        As process char may abort early, provide final evaluation full count
        '''
        if count == 0:
            return TypeResolved.IS_EMPTY
        if self.resolved_field != TypeResolved.IS_UNKNOWN:
            return self.resolved_field
        # determine
        if self.contiguous_numeric:
            # NOTE: have already handled cases with excessive counts
            if self.count_j == 0 and self.count_e == 0 and self.count_decimal == 0:
                return TypeResolved.IS_INT
            if self.count_j == 0 and (self.count_decimal == 1 or self.count_e > 0):
                return TypeResolved.IS_FLOAT
            if self.count_j == 1:
                return TypeResolved.IS_COMPLEX
        return TypeResolved.IS_STRING

    @staticmethod
    def resolve_line_type(previous: TypeResolved, new: TypeResolved) -> None:
        if previous is TypeResolved.IS_UNKNOWN:
            return new

        # a string with anything else is a string
        if previous is TypeResolved.IS_STRING or new is TypeResolved.IS_STRING:
            return TypeResolved.IS_STRING

        if previous is TypeResolved.IS_BOOL:
            if new is TypeResolved.IS_BOOL:
                return TypeResolved.IS_BOOL
            else: # bool found with anything else is a string
                return TypeResolved.IS_STRING

        if previous is TypeResolved.IS_INT:
            if (new is TypeResolved.IS_EMPTY
                    or new is TypeResolved.IS_INT):
                return TypeResolved.IS_INT
            if new is TypeResolved.IS_FLOAT:
                return TypeResolved.IS_FLOAT
            if new is TypeResolved.IS_COMPLEX:
                return TypeResolved.IS_COMPLEX

        if previous is TypeResolved.IS_FLOAT:
            if (new is TypeResolved.IS_EMPTY
                    or new is TypeResolved.IS_INT
                    or new is TypeResolved.IS_FLOAT):
                return TypeResolved.IS_FLOAT
            if new is TypeResolved.IS_COMPLEX:
                return TypeResolved.IS_COMPLEX

        if previous is TypeResolved.IS_COMPLEX:
            if (new is TypeResolved.IS_EMPTY
                    or new is TypeResolved.IS_INT
                    or new is TypeResolved.IS_FLOAT
                    or new is TypeResolved.IS_COMPLEX):
                return TypeResolved.IS_COMPLEX

        raise NotImplementedError(previous, new)

    def process(self, field: str) -> TypeResolved:
        print(f'process: {field=}')

        self.reset() # does not reset resolved_line
        pos = 0
        continue_process = 1
        for char in field:
            if continue_process:
                continue_process = self.process_char(char, pos)
            pos += 1 # results in count

        # must call after all chars processed, does not set self.resolved field
        rlt_new = self.resolve_field_type(pos)
        self.resolved_line = self.resolve_line_type(self.resolved_line, rlt_new)
        print(f'{self.resolved_line=}')
        return self.resolved_line

    def process_line(self, fields: tp.Iterable[str]) -> TypeResolved:
        for field in fields:
            self.process(field)
        return self.resolved_line

class TestUnit(unittest.TestCase):

    def test_bool_a(self) -> None:
        self.assertEqual(TypeField().process('   true'), TypeResolved.IS_BOOL)
        self.assertEqual(TypeField().process('FALSE'), TypeResolved.IS_BOOL)
        self.assertEqual(TypeField().process('  tals  '), TypeResolved.IS_STRING)


    def test_str_a(self) -> None:
        self.assertEqual(TypeField().process('+++'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process('   ee   '), TypeResolved.IS_STRING)

    def test_int_a(self) -> None:
        self.assertEqual(TypeField().process(' 3'), TypeResolved.IS_INT)
        self.assertEqual(TypeField().process('3 '), TypeResolved.IS_INT)
        self.assertEqual(TypeField().process('  +3 '), TypeResolved.IS_INT)
        self.assertEqual(TypeField().process('+599w'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process('k599'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process('59 4'), TypeResolved.IS_STRING)

        self.assertEqual(TypeField().process('153'), TypeResolved.IS_INT)
        self.assertEqual(TypeField().process('  153  '), TypeResolved.IS_INT)
        self.assertEqual(TypeField().process('  15 3'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process('5 3'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process(' 5 3 '), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process('  5 3 '), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process('  5  3 '), TypeResolved.IS_STRING)

    def test_float_a(self) -> None:
        self.assertEqual(TypeField().process(' .3'), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process('3. '), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process(' 2343. '), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process(' 2343.9 '), TypeResolved.IS_FLOAT)

        self.assertEqual(TypeField().process(' 23t3.9 '), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process(' 233.9!'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process('4.3.5'), TypeResolved.IS_STRING)

    def test_float_b(self) -> None:
        self.assertEqual(TypeField().process(' 4e3'), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process(' 4e3e'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process('4e3   e'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process('e99   '), TypeResolved.IS_STRING)

    def test_float_known_false_positive(self) -> None:
        # NOTE: we mark this as float because we do not observe that a number must follow e; assume this will fail in float conversion
        self.assertEqual(TypeField().process('8e'), TypeResolved.IS_FLOAT)

    def test_complex_a(self) -> None:
        self.assertEqual(TypeField().process('23j  '), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process(' 4e3j'), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process(' 4e3jw'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process(' J4e3j'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process('-4.3+3j'), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process(' j4e3'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process('j11111    '), TypeResolved.IS_STRING)

    def test_complex_b(self) -> None:
        self.assertEqual(TypeField().process('2.3-3.5j  '), TypeResolved.IS_COMPLEX)

    def test_float_known_false_positive(self) -> None:
        # NOTE: genfromtxt identifies this as string as j component is in first position
        self.assertEqual(TypeField().process('23j-43'), TypeResolved.IS_COMPLEX)


    def test_line_a(self) -> None:
        self.assertEqual(TypeField().process_line(('25', '2.5', '')), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_line((' .1', '2.5', '')), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_line(('25', '', '')), TypeResolved.IS_INT)



if __name__ == '__main__':
    unittest.main()



