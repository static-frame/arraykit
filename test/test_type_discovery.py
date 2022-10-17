



import unittest
import typing as tp
from enum import Enum

from hypothesis import strategies as st
from hypothesis import given


_ = '''
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

def is_paren_open(c: str) -> bool:
    return c == '('

def is_paren_close(c: str) -> bool:
    return c == ')'

def is_decimal(c: str) -> bool:
    return c == '.'


def is_a(c: str) -> bool:
    return c == 'a' or c == 'A'

def is_e(c: str) -> bool:
    return c == 'e' or c == 'E'

def is_f(c: str) -> bool:
    return c == 'f' or c == 'F'

def is_i(c: str) -> bool:
    return c == 'i' or c == 'I'

def is_j(c: str) -> bool:
    return c == 'j' or c == 'J'

def is_l(c: str) -> bool:
    return c == 'l' or c == 'L'

def is_n(c: str) -> bool:
    return c == 'n' or c == 'N'

def is_r(c: str) -> bool:
    return c == 'r' or c == 'R'

def is_s(c: str) -> bool:
    return c == 's' or c == 'S'

def is_t(c: str) -> bool:
    return c == 't' or c == 'T'

def is_u(c: str) -> bool:
    return c == 'u' or c == 'U'


class TypeResolved(Enum):
    IS_UNKNOWN = 1
    IS_BOOL = 2
    IS_INT = 3
    IS_FLOAT = 4
    IS_COMPLEX = 5
    IS_STRING = 6
    IS_EMPTY = 7

    @classmethod
    def resolve(cls, previous: 'TypeResolved', new: 'TypeResolved') -> None:
        if new is cls.IS_UNKNOWN:
            return cls.IS_STRING

        if (previous is cls.IS_UNKNOWN
                or previous is cls.IS_EMPTY):
            return new

        # a string with anything else is a string
        if (previous is cls.IS_STRING
                or new is cls.IS_STRING):
            return cls.IS_STRING

        if previous is cls.IS_BOOL:
            if (new is cls.IS_EMPTY
                    or new is cls.IS_BOOL):
                return cls.IS_BOOL
            else: # bool found with anything else except empty is a string
                return cls.IS_STRING
        if new is cls.IS_BOOL:
            if previous is cls.IS_EMPTY:
                return cls.IS_BOOL
            else:
                return cls.IS_STRING


        if previous is cls.IS_INT:
            if (new is cls.IS_EMPTY
                    or new is cls.IS_INT):
                return cls.IS_INT
            if new is cls.IS_FLOAT:
                return cls.IS_FLOAT
            if new is cls.IS_COMPLEX:
                return cls.IS_COMPLEX

        if previous is cls.IS_FLOAT:
            if (new is cls.IS_EMPTY
                    or new is cls.IS_INT
                    or new is cls.IS_FLOAT):
                return cls.IS_FLOAT
            if new is cls.IS_COMPLEX:
                return cls.IS_COMPLEX

        # if previous is cls.IS_COMPLEX:
        #     if (new is cls.IS_EMPTY
        #             or new is cls.IS_INT
        #             or new is cls.IS_FLOAT
        #             or new is cls.IS_COMPLEX):
        return cls.IS_COMPLEX

        raise NotImplementedError(previous, new)

class TypeField:
    '''
    Estimate the type of a field. This estimate can be based on character type counts. Some ordering considerations will be ignored for convenience; if downstream parsing fails, fallback will be to a string type anyway.
    '''
    def __init__(self) -> None:
        self.reset()
        self.parsed_line: TypeResolved = TypeResolved.IS_UNKNOWN

    def reset(self) -> None:
        self.parsed_field: TypeResolved = TypeResolved.IS_UNKNOWN

        self.previous_numeric = False
        self.contiguous_leading_space = False
        self.contiguous_numeric = False

        # numeric symbols; values do not need to be greater than 8
        self.count_bool = 0 # signed, not greater than +/- 5
        self.count_sign = 0
        self.count_e = 0
        self.count_j = 0
        self.count_decimal = 0
        self.count_nan = 0
        self.count_inf = 0
        self.count_paren_open = 0
        self.count_paren_close = 0

        # can be unbound in size
        self.last_sign_pos = -1
        self.count_leading_space = 0
        self.count_digit = 0
        self.count_notspace = 0 # non-space, non-paren

    def process_char(self, c: str, pos: int) -> int:
        # position is postion needs to be  dropping leading space
        # update self based on c and position
        # return int where 1 means process more, 0 means stop, -1 means error

        if self.parsed_field != TypeResolved.IS_UNKNOWN:
            return 0

         # 32 to 57; 65 to 85; 97 to 117 inclusive
        # if ord(c) < 32 or ord(c) > 117: # less than space, greater than u
        #     self.parsed_field = TypeResolved.IS_STRING
        #     return 0
        # if 57 < ord(c) < 65: # greater than 9, less than A
        #     self.parsed_field = TypeResolved.IS_STRING
        #     return 0
        # if 85 < ord(c) < 97: # greater than U, less than a
        #     self.parsed_field = TypeResolved.IS_STRING
        #     return 0

        # evaluate space -------------------------------------------------------
        space = False

        if is_space(c):
            if pos == 0:
                self.contiguous_leading_space = True
            if self.contiguous_leading_space:
                self.count_leading_space += 1
                return 1
            space = True
        elif is_paren_open(c):
            self.count_paren_open += 1
            self.count_leading_space += 1
            space = True
            # open paren only permitted first non-space position
            if (pos > 0 and not self.contiguous_leading_space) or self.count_paren_open > 1:
                self.parsed_field = TypeResolved.IS_STRING
                return 0
        elif is_paren_close(c):
            self.count_paren_close += 1
            space = True
            # # NOTE: not evaluating that this is on last position of contiguous numeric
            if self.count_paren_close > 1:
                self.parsed_field = TypeResolved.IS_STRING
                return 0
        else:
            self.count_notspace += 1

        self.contiguous_leading_space = False

        pos_field = pos - self.count_leading_space

        # evaluate numeric, non-positional -------------------------------------
        numeric = False
        digit = False

        if space:
            pass

        elif is_digit(c):
            numeric = True
            digit = True
            self.count_digit += 1

        elif is_decimal(c):
            self.count_decimal += 1
            if self.count_decimal > 2: # complex can have 2!
                self.parsed_field = TypeResolved.IS_STRING
                return 0
            numeric = True

        elif is_sign(c):
            self.count_sign += 1
            if self.count_sign > 4:
                # complex numbers with E can have up to 4 signs, anything else is a string
                self.parsed_field = TypeResolved.IS_STRING
                return 0
            self.last_sign_pos = pos_field
            numeric = True


        elif is_e(c): # only character that is numeric and bool
            self.count_e += 1
            if pos_field == 0 or self.count_e > 2:
                # true and false each only have one E, complex can have 2
                self.parsed_field = TypeResolved.IS_STRING
                return 0
            numeric = True

        elif is_j(c):
            self.count_j += 1
            if pos_field == 0 or self.count_j > 1:
                self.parsed_field = TypeResolved.IS_STRING
                return 0
            numeric = True


        #-----------------------------------------------------------------------
        # print(f' pre: {c=} {pos=} {pos_field=} {numeric=} {self.previous_numeric=} {self.contiguous_numeric=}')

        if numeric:
            if pos_field == 0:
                self.contiguous_numeric = True
                self.previous_numeric = True
            # pos_field > 0
            if not self.previous_numeric:
                # found a numeric not in pos 0 where previous was not numeric
                self.contiguous_numeric = False
            self.previous_numeric = True

        else: # not numeric, could be space or notspace
            if self.contiguous_numeric and not space:
                self.contiguous_numeric = False

            self.previous_numeric = False


        # evaluate character positions -----------------------------------------
        if space or digit:
            return 1

        # if we have a last sign, it takes precedence over use count_paren_open as a shfit
        if self.last_sign_pos >= 0:
            pos_field -= self.last_sign_pos + 1

        if pos_field == 0:
            if is_t(c):
                self.count_bool += 1
            elif is_f(c):
                self.count_bool -= 1
            elif is_n(c):
                self.count_nan += 1
            elif is_i(c):
                self.count_inf += 1
            elif not numeric:
                self.parsed_field = TypeResolved.IS_STRING
                return 0

        elif pos_field == 1:
            if is_r(c):
                self.count_bool += 1
            elif is_a(c):
                self.count_bool -= 1
                self.count_nan += 1
            elif is_n(c):
                self.count_inf += 1
            elif not numeric:
                self.parsed_field = TypeResolved.IS_STRING
                return 0

        elif pos_field == 2:
            if is_u(c):
                self.count_bool += 1
            elif is_l(c):
                self.count_bool -= 1
            elif is_n(c):
                self.count_nan += 1
            elif is_f(c):
                self.count_inf += 1
            elif not numeric:
                self.parsed_field = TypeResolved.IS_STRING
                return 0

        elif pos_field == 3:
            if is_e(c):
                self.count_bool += 1
            if is_s(c):
                self.count_bool -= 1
            elif not numeric:
                self.parsed_field = TypeResolved.IS_STRING
                return 0

        elif pos_field == 4:
            if is_e(c):
                self.count_bool -= 1
            elif not numeric:
                self.parsed_field = TypeResolved.IS_STRING
                return 0

        elif not numeric:
            self.parsed_field = TypeResolved.IS_STRING
            return 0

        # print(f'post: {c=} {pos=} {pos_field=} {numeric=} {self.previous_numeric=} {self.contiguous_numeric=} {self.last_sign_pos=} {self.count_nan=} {self.count_inf=} {self.count_notspace=}')


        return 1

    def resolve_field_type(self, count: int) -> None:
        '''
        As process char may abort early, provide final evaluation full count
        '''
        if count == 0:
            return TypeResolved.IS_EMPTY

        if self.parsed_field != TypeResolved.IS_UNKNOWN:
            return self.parsed_field

        if self.count_bool == 4 and self.count_notspace == 4:
            return TypeResolved.IS_BOOL
        if self.count_bool == -5 and self.count_notspace == 5:
            return TypeResolved.IS_BOOL

        if self.contiguous_numeric: # must have digits
            # NOTE: have already handled cases with excessive counts
            if self.count_digit == 0:
                # can have contiguous numerics like +ej.- but no digits
                return TypeResolved.IS_STRING

            if (self.count_j == 0
                    and self.count_e == 0
                    and self.count_decimal == 0
                    and self.count_paren_open == 0
                    and self.count_paren_close == 0
                    and self.count_nan == 0
                    and self.count_inf == 0):
                return TypeResolved.IS_INT

            if (self.count_j == 0
                    and self.count_sign <= 2
                    and self.count_paren_open == 0
                    and self.count_paren_close == 0
                    and (self.count_decimal == 1 or self.count_e == 1)):
                if self.count_sign > 1 and self.count_e == 0:
                    # if more than one sign and no e, not a float
                    return TypeResolved.IS_STRING
                return TypeResolved.IS_FLOAT

            if self.count_j == 1 and (
                    (self.count_paren_open == 1 and self.count_paren_close == 1)
                    or (self.count_paren_open == 0 and self.count_paren_close == 0)
                    ):
                if self.count_sign > 2 + self.count_e:
                    return TypeResolved.IS_STRING
                return TypeResolved.IS_COMPLEX

            # if only paren and digits, mark as complex
            if self.count_j == 0 and (
                    (self.count_paren_open == 1 and self.count_paren_close == 1)
                    ):
                if self.count_e > 1 or self.count_sign > 2:
                    return TypeResolved.IS_STRING
                return TypeResolved.IS_COMPLEX

        # not contiguous numeric, has inf or nan in some combination
        elif self.count_j == 0:
            # float nan and inf that might be signed
            if self.count_nan == 3 and self.count_sign + self.count_nan == self.count_notspace:
                return TypeResolved.IS_FLOAT
            if self.count_inf == 3 and self.count_sign + self.count_inf == self.count_notspace:
                return TypeResolved.IS_FLOAT

        elif self.count_j == 1:
            # special cases of complex that do not present as contiguous numeric because of inf/nan
            if self.count_inf == 3 or self.count_inf == 6 and (
                    self.count_sign + self.count_inf + 1 == self.count_notspace
                    ):
                return TypeResolved.IS_COMPLEX
            if self.count_nan == 3 or self.count_nan == 6 and (
                    self.count_sign + self.count_nan + 1 == self.count_notspace
                    ):
                return TypeResolved.IS_COMPLEX


            # import ipdb; ipdb.set_trace()

        return TypeResolved.IS_STRING


    def process_field(self, field: str) -> TypeResolved:
        # NOTE: return TypeResolved is not necessary

        self.reset() # does not reset parsed_line
        pos = 0
        continue_process = 1
        for char in field:
            if continue_process:
                continue_process = self.process_char(char, pos)
            pos += 1 # results in count

        # must call after all chars processed, does not set self.resolved field
        rlt_new = self.resolve_field_type(pos)
        self.parsed_line = TypeResolved.resolve(self.parsed_line, rlt_new)
        # print(f'{self.parsed_line=}')
        return self.parsed_line # returning this is just for testing

    def get_resolved(self) -> TypeResolved:
        if self.parsed_line is TypeResolved.IS_EMPTY:
            return TypeResolved.IS_FLOAT
        if self.parsed_line is TypeResolved.IS_UNKNOWN:
            return TypeResolved.IS_STRING
        return self.parsed_line

    def process_line(self, fields: tp.Iterable[str]) -> TypeResolved:
        for field in fields:
            self.process_field(field)
        return self.get_resolved()


class TestUnit(unittest.TestCase):

    def test_bool_a(self) -> None:
        self.assertEqual(TypeField().process_field('   true'), TypeResolved.IS_BOOL)
        self.assertEqual(TypeField().process_field('FALSE'), TypeResolved.IS_BOOL)
        self.assertEqual(TypeField().process_field('FaLSE   '), TypeResolved.IS_BOOL)

        self.assertEqual(TypeField().process_field('  tals  '), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('FALSEblah'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('   true f'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('   true3'), TypeResolved.IS_STRING)

    def test_bool_b(self) -> None:
        self.assertEqual(TypeField().process_field('   true +'), TypeResolved.IS_STRING)


    def test_str_a(self) -> None:
        self.assertEqual(TypeField().process_field('+++'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('   ee   '), TypeResolved.IS_STRING)


    @given(st.integers())
    def test_int_property(self, v) -> None:
        self.assertEqual(TypeField().process_field(str(v)), TypeResolved.IS_INT)

    def test_int_a(self) -> None:
        self.assertEqual(TypeField().process_field(' 3'), TypeResolved.IS_INT)
        self.assertEqual(TypeField().process_field('3 '), TypeResolved.IS_INT)
        self.assertEqual(TypeField().process_field('  +3 '), TypeResolved.IS_INT)
        self.assertEqual(TypeField().process_field('+599w'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('k599'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('59 4'), TypeResolved.IS_STRING)

        self.assertEqual(TypeField().process_field('153'), TypeResolved.IS_INT)
        self.assertEqual(TypeField().process_field('  153  '), TypeResolved.IS_INT)
        self.assertEqual(TypeField().process_field('  15 3'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('5 3'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field(' 5 3 '), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('  5 3 '), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('  5  3 '), TypeResolved.IS_STRING)

    @given(st.floats())
    def test_float_property(self, v) -> None:
        self.assertEqual(TypeField().process_field(str(v)), TypeResolved.IS_FLOAT)

    def test_float_a(self) -> None:
        self.assertEqual(TypeField().process_field(' .3'), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_field('3. '), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_field(' 2343. '), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_field(' 2343.9 '), TypeResolved.IS_FLOAT)

        self.assertEqual(TypeField().process_field(' 23t3.9 '), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field(' 233.9!'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('4.3.5'), TypeResolved.IS_STRING)

    def test_float_b(self) -> None:
        self.assertEqual(TypeField().process_field(' 4e3'), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_field('4E3 '), TypeResolved.IS_FLOAT)

        self.assertEqual(TypeField().process_field(' 4e3e'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('4e3   e'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('e99   '), TypeResolved.IS_STRING)

    def test_float_c(self) -> None:
        self.assertEqual(TypeField().process_field('  .  '), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('..'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('e+j.'), TypeResolved.IS_STRING)

    def test_float_d(self) -> None:
        self.assertEqual(TypeField().process_field('  nan'), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_field('NaN   '), TypeResolved.IS_FLOAT)

        self.assertEqual(TypeField().process_field('NaN3   '), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field(' N an   '), TypeResolved.IS_STRING)

    def test_float_e(self) -> None:
        self.assertEqual(TypeField().process_field('-inf'), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_field('inf'), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_field('INF   '), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_field(' +InF   '), TypeResolved.IS_FLOAT)

    def test_float_f(self) -> None:
        self.assertEqual(TypeField().process_field('-nan'), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_field('nan'), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_field('nan   '), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_field(' +nan   '), TypeResolved.IS_FLOAT)

    def test_float_g(self) -> None:
        self.assertEqual(TypeField().process_field('8.++'), TypeResolved.IS_STRING)

    def test_float_known_false_positive(self) -> None:
        # NOTE: we mark this as float because we do not observe that a number must follow e; assume this will fail in float conversion
        self.assertEqual(TypeField().process_field('8e'), TypeResolved.IS_FLOAT)


    @given(st.complex_numbers())
    def test_complex_property(self, v) -> None:
        # print(v, TypeField().process_field(str(v)))
        self.assertEqual(TypeField().process_field(str(v)), TypeResolved.IS_COMPLEX)

    def test_complex_a(self) -> None:
        self.assertEqual(TypeField().process_field('23j  '), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field(' 4e3j'), TypeResolved.IS_COMPLEX)

        self.assertEqual(TypeField().process_field(' 4e3jw'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field(' J4e3j'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('-4.3+3j'), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field(' j4e3'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('j11111    '), TypeResolved.IS_STRING)

    def test_complex_b(self) -> None:
        self.assertEqual(TypeField().process_field('2.3-3.5j  '), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field('+23-35j  '), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field('+23-3.5j  '), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field('-3e-10-3e-2j'), TypeResolved.IS_COMPLEX)

        self.assertEqual(TypeField().process_field('+23-3.5j  +'), TypeResolved.IS_STRING)

    def test_complex_c(self) -> None:
        self.assertEqual(TypeField().process_field(' (23+3j) '), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field('(4e3-4.5j)'), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field('(4.3)'), TypeResolved.IS_COMPLEX)

        self.assertEqual(TypeField().process_field(' (23+3j)) '), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field(' (((23+3j'), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field(' 2(3+3j) '), TypeResolved.IS_STRING)
        self.assertEqual(TypeField().process_field('(23+)3j '), TypeResolved.IS_STRING)

    def test_complex_d(self) -> None:
        self.assertEqual(TypeField().process_field(' infj'), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field(' -infj'), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field(' +infj'), TypeResolved.IS_COMPLEX)

        self.assertEqual(TypeField().process_field(' nanj'), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field(' -nanj  '), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field(' +nanj'), TypeResolved.IS_COMPLEX)

    def test_complex_e(self) -> None:
        self.assertEqual(TypeField().process_field(' inf+0j'), TypeResolved.IS_COMPLEX)

    def test_complex_f(self) -> None:
        self.assertEqual(TypeField().process_field(' inf+nanj'), TypeResolved.IS_COMPLEX)

    def test_complex_g(self) -> None:
        self.assertEqual(TypeField().process_field(' inf-infj'), TypeResolved.IS_COMPLEX)

    def test_complex_h(self) -> None:
        self.assertEqual(TypeField().process_field(' -0+infj'), TypeResolved.IS_COMPLEX)

    def test_complex_i(self) -> None:
        self.assertEqual(TypeField().process_field('(inf+0j)'), TypeResolved.IS_COMPLEX)

    def test_complex_j1(self) -> None:
        self.assertEqual(TypeField().process_field('(-0+infj)'), TypeResolved.IS_COMPLEX)

    def test_complex_j2(self) -> None:
        self.assertEqual(TypeField().process_field('(-23e-10e)'), TypeResolved.IS_STRING)


    def test_complex_k(self) -> None:
        self.assertEqual(TypeField().process_field('(-23e-10j-34e-2)'), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field('(-23e-10j-34e-2+)'), TypeResolved.IS_STRING)


    def test_complex_known_false_positive(self) -> None:
        # NOTE: genfromtxt identifies this as string as j component is in first position
        self.assertEqual(TypeField().process_field('23j-43'), TypeResolved.IS_COMPLEX)
        self.assertEqual(TypeField().process_field('+23-3.5j3'), TypeResolved.IS_COMPLEX)




    def test_line_a(self) -> None:
        self.assertEqual(TypeField().process_line(('25', '2.5', '')), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_line((' .1', '2.5', '')), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_line(('25', '', '')), TypeResolved.IS_INT)

        self.assertEqual(TypeField().process_line(('25', '2.5', 'e')), TypeResolved.IS_STRING)

    def test_line_b(self) -> None:
        self.assertEqual(TypeField().process_line(('  true', '  false', 'FALSE')), TypeResolved.IS_BOOL)
        self.assertEqual(TypeField().process_line(('  true', '  false', 'FALSEq')), TypeResolved.IS_STRING)

    def test_line_c(self) -> None:
        self.assertEqual(TypeField().process_line(('3', '', '4')), TypeResolved.IS_INT)

        self.assertEqual(TypeField().process_line(('3', '', '4e')), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_line(('3', '', '.')), TypeResolved.IS_STRING)

        self.assertEqual(TypeField().process_line(('', '', '')), TypeResolved.IS_FLOAT)

    def test_line_d(self) -> None:
        self.assertEqual(TypeField().process_line(('3', '', '4.')), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_line(('3', '', '4e3')), TypeResolved.IS_FLOAT)
        self.assertEqual(TypeField().process_line(('3', '', '(4e3)')), TypeResolved.IS_COMPLEX)

        self.assertEqual(TypeField().process_line(('3', '', '(4e3)', 'True')), TypeResolved.IS_STRING)


    def test_line_e(self) -> None:
        self.assertEqual(TypeField().process_line(('foo', '', '', 'bar')), TypeResolved.IS_STRING)

        self.assertEqual(TypeField().process_line(('', '', '', 'bar')), TypeResolved.IS_STRING)

    def test_line_f(self) -> None:
        # EMPTY is treated as False
        self.assertEqual(TypeField().process_line(('', '', '', 'True')), TypeResolved.IS_BOOL)

        self.assertEqual(TypeField().process_line(('True', '')), TypeResolved.IS_BOOL)



if __name__ == '__main__':
    unittest.main()



