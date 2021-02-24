
import unittest
import copy
import pickle

import numpy as np

from arraykit import ArrayGO
from arraykit import mloc

class TestUnit(unittest.TestCase):


    def test_array_init_a(self) -> None:
        with self.assertRaises(NotImplementedError):
            _ = ArrayGO(np.array((3, 4, 5)))

    def test_array_append_a(self) -> None:

        ag1 = ArrayGO(('a', 'b', 'c', 'd'))

        self.assertEqual([x for x in ag1],
            ['a', 'b', 'c', 'd'])

        self.assertEqual(ag1.values.tolist(),
                ['a', 'b', 'c', 'd'])


        ag1.append('e')
        ag1.extend(('f', 'g'))

        self.assertEqual(ag1.values.tolist(),
            ['a', 'b', 'c', 'd', 'e', 'f', 'g'])

        self.assertEqual([x for x in ag1],
            ['a', 'b', 'c', 'd', 'e', 'f', 'g'])


    def test_array_append_b(self) -> None:

        ag1 = ArrayGO(np.array(('a', 'b', 'c', 'd'), object))

        self.assertEqual([x for x in ag1],
            ['a', 'b', 'c', 'd'])

        self.assertEqual(ag1.values.tolist(),
                ['a', 'b', 'c', 'd'])


        ag1.append('e')
        ag1.extend(('f', 'g'))

        self.assertEqual(ag1.values.tolist(),
            ['a', 'b', 'c', 'd', 'e', 'f', 'g'])

        self.assertEqual([x for x in ag1],
            ['a', 'b', 'c', 'd', 'e', 'f', 'g'])


    def test_array_getitem_a(self) -> None:

        a = np.array(('a', 'b', 'c', 'd'), object)
        a.flags.writeable = False

        ag1 = ArrayGO(a)
        # insure no copy for immutable
        self.assertEqual(mloc(ag1.values), mloc(a))

        ag1.append('b')

        post = ag1[ag1.values == 'b']

        self.assertEqual(post.tolist(), ['b', 'b'])
        self.assertEqual(ag1[[2,1,1,1]].tolist(),
                ['c', 'b', 'b', 'b'])

    def test_array_copy_a(self) -> None:

        ag1 = ArrayGO(np.array(('a', 'b', 'c', 'd'), dtype=object))
        ag1.append('e')

        ag2 = ag1.copy()
        ag1.extend(('f', 'g'))

        self.assertEqual(ag1.values.tolist(),
                ['a', 'b', 'c', 'd', 'e', 'f', 'g'])

        self.assertEqual(ag2.values.tolist(),
                ['a', 'b', 'c', 'd', 'e'])

    def test_array_deepcopy_a(self) -> None:
        ag1 = ArrayGO(np.array(('a', 'b', 'c', 'd'), dtype=object))
        ag1.append('e')
        ag1.extend(('f', 'g'))
        ag2 = copy.deepcopy(ag1)
        self.assertEqual(ag1.values.tolist(), ag2.values.tolist()) #type: ignore

    def test_array_len_a(self) -> None:

        ag1 = ArrayGO(np.array(('a', 'b', 'c', 'd'), object))
        ag1.append('e')

        self.assertEqual(len(ag1), 5)

    def test_array_getnewargs_a(self) -> None:
        ag1 = ArrayGO(np.array(('a', 'b', 'c', 'd'), object))
        self.assertEqual(
                ag1.__getnewargs__()[0].tolist(),
                ag1.values.tolist(),
                )

    def test_array_pickle_a(self) -> None:
        ag1 = ArrayGO(np.array(('a', 'b', 'c', 'd'), object))
        msg = pickle.dumps(ag1)
        ag2 = pickle.loads(msg)
        self.assertEqual(ag1.values.tolist(), ag2.values.tolist())



if __name__ == '__main__':
    unittest.main()




