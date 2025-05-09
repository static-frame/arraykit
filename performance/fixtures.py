import typing as tp


import numpy as np

from arraymap import AutoMap
from arraymap import FrozenAutoMap


class PayLoad:
    def __init__(self, array: np.ndarray):
        self.array = array
        self.list = list(array)
        self.faml = FrozenAutoMap(self.list)
        self.fama = FrozenAutoMap(self.array)
        self.ama = AutoMap(self.array)
        self.d = dict(zip(self.list, range(len(self.list))))
        self.sel_array = array[(np.arange(len(array)) % 2) == 0]
        self.sel_scalar = list(self.sel_array)


# -------------------------------------------------------------------------------
INT_START = 500  # avoid cached ints starting at 256


class FixtureFactory:
    NAME = ""
    SORT = 0
    CACHE = {}  # can be shared for all classes

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, PayLoad]:
        key = (cls, size)
        if key not in cls.CACHE:
            pl = PayLoad(cls.get_array(size))
            cls.CACHE[key] = pl
        return cls.NAME, cls.CACHE[key]


class FFInt64(FixtureFactory):
    NAME = "int64"
    SORT = 0

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(INT_START, INT_START + size, dtype=np.int64)
        array.flags.writeable = False
        return array


class FFInt32(FixtureFactory):
    NAME = "int32"
    SORT = 1

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(INT_START, INT_START + size, dtype=np.int32)
        array.flags.writeable = False
        return array


class FFUInt64(FixtureFactory):
    NAME = "uint64"
    SORT = 2

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(INT_START, INT_START + size, dtype=np.uint64)
        array.flags.writeable = False
        return array


class FFUInt32(FixtureFactory):
    NAME = "uint32"
    SORT = 3

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(INT_START, INT_START + size, dtype=np.uint32)
        array.flags.writeable = False
        return array


class FFFloat64(FixtureFactory):
    NAME = "float64"
    SORT = 4

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = (np.arange(INT_START, INT_START + size) * 0.5).astype(np.float64)
        array.flags.writeable = False
        return array


class FFFloat32(FixtureFactory):
    NAME = "float32"
    SORT = 5

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = (np.arange(INT_START, INT_START + size) * 0.5).astype(np.float32)
        array.flags.writeable = False
        return array


def get_string_array(size: int, char_count: int, kind: str) -> str:
    fmt = f"-<{char_count}"
    array = np.array(
        [
            f"{hex(e) * (char_count // 8)}".format(fmt)
            for e in range(INT_START, INT_START + size)
        ],
        dtype=f"{kind}{char_count}",
    )
    array.flags.writeable = False
    return array


class FFU8(FixtureFactory):
    NAME = "U8"
    SORT = 6

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return get_string_array(size, 8, "U")


class FFU16(FixtureFactory):
    NAME = "U16"
    SORT = 7

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return get_string_array(size, 16, "U")


class FFU32(FixtureFactory):
    NAME = "U32"
    SORT = 8

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return get_string_array(size, 32, "U")


class FFU64(FixtureFactory):
    NAME = "U64"
    SORT = 9

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return get_string_array(size, 64, "U")


class FFU128(FixtureFactory):
    NAME = "U128"
    SORT = 10

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return get_string_array(size, 128, "U")


class FFS8(FixtureFactory):
    NAME = "S8"
    SORT = 11

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return get_string_array(size, 8, "S")


class FFS16(FixtureFactory):
    NAME = "S16"
    SORT = 12

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return get_string_array(size, 16, "S")


class FFS32(FixtureFactory):
    NAME = "S32"
    SORT = 13

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return get_string_array(size, 32, "S")


class FFS64(FixtureFactory):
    NAME = "S64"
    SORT = 14

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return get_string_array(size, 64, "S")


class FFS128(FixtureFactory):
    NAME = "S128"
    SORT = 15

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        return get_string_array(size, 128, "S")


class FFDTY(FixtureFactory):
    NAME = "dt[Y]"
    SORT = 20

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(INT_START, INT_START + size, dtype="datetime64[Y]")
        array.flags.writeable = False
        return array


class FFDTD(FixtureFactory):
    NAME = "dt[D]"
    SORT = 21

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(INT_START, INT_START + size, dtype="datetime64[D]")
        array.flags.writeable = False
        return array


class FFDTs(FixtureFactory):
    NAME = "dt[s]"
    SORT = 22

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(INT_START, INT_START + size, dtype="datetime64[s]")
        array.flags.writeable = False
        return array


class FFDTns(FixtureFactory):
    NAME = "dt[ns]"
    SORT = 23

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(INT_START, INT_START + size, dtype="datetime64[ns]")
        array.flags.writeable = False
        return array


class FFObject(FixtureFactory):
    NAME = "object"
    SORT = 5

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        ints = np.arange(INT_START, INT_START + size)
        array = ints.astype(object)

        target = 1 == ints % 3
        array[target] = ints[target] * 0.5

        target = 2 == ints % 3
        array[target] = np.array([hex(e) for e in ints[target]])

        array.flags.writeable = False
        return array
