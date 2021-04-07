from src.pre_processing import Node
from src.config import Config
import unittest


class NodeTests(unittest.TestCase):
    def test_expand_struct(self):
        Config.sequence_lengths = [4, 4, 4, 4]

        root = Node(level=1)
        root.expand_struct(struct=[[10, 11, 12, 13, 2]])
        self.assertEqual(
            '[[10, 11, 12, 13], [2]]',
            str(root.build_struct())
        )

        root = Node(level=2)
        root.expand_struct(struct=[[[10, 11, 12, 13, 2], [14, 15, 16, 2], [17, 18, 19, 20, 21, 22, 2]]])
        self.assertEqual(
            '[[[10, 11, 12, 13], [2], [14, 15, 16, 2]], [[17, 18, 19, 20], [21, 22, 2]]]',
            str(root.build_struct())
        )
