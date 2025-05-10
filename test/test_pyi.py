import typing as tp
import os
import unittest

from importlib.util import spec_from_loader
from importlib.util import module_from_spec


import arraykit as ak

class Interface(tp.NamedTuple):
    functions: tp.List[str]
    classes: tp.Dict[str, tp.List[str]]

    @staticmethod
    def _valid_name(name: str) -> bool:
        if name.startswith('__'):
            return True
        if name.startswith('_'):
            return False
        return True

    @classmethod
    def from_module(cls, module):
        functions: tp.List[str] = []
        classes: tp.Dict[str: tp.List[str]] = {}

        for name in dir(module):
            if not cls._valid_name(name):
                continue
            obj = getattr(module, name)
            if isinstance(obj, type): # a class
                if name in (ak.ErrorInitTypeBlocks.__name__, ak.NonUniqueError.__name__):
                    # skip as there is Python version variability
                    continue
                classes[name] = []
                for part_name in dir(obj):
                    if not cls._valid_name(part_name):
                        continue
                    part_obj = getattr(obj, part_name)
                    if callable(part_obj):
                        classes[name].append(part_name)
            elif callable(obj):
                functions.append(name)

        return cls(functions, classes)


class TestUnit(unittest.TestCase):

    # @unittest.skip('not sure if pyi is in right location')
    def test_interface(self) -> None:

        fp = os.path.join(os.path.dirname(ak.__file__), '__init__.pyi')

        with open(fp) as f:
            msg = f.read()

        spec = spec_from_loader('', loader=None)
        pyi_mod = module_from_spec(spec)

        exec(msg, pyi_mod.__dict__)

        ak_content = Interface.from_module(ak)
        pyi_content = Interface.from_module(pyi_mod)

        self.assertEqual(ak_content.functions, pyi_content.functions)

        for name in ak_content.classes.keys():
            ak_class = ak_content.classes[name]
            pyi_class = pyi_content.classes[name]

            if '__hash__' in ak_class: ak_class.remove('__hash__')
            if '__hash__' in pyi_class: pyi_class.remove('__hash__')

            self.assertEqual(ak_class, pyi_class)


if __name__ == '__main__':
    unittest.main()
