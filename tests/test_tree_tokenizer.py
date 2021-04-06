from src.pre_processing import TreeTokenizer
import unittest


class TreeTokenizerTests(unittest.TestCase):
    dummy_expected = ['I like big butts.<s>I can not lie.', 'some other song.']

    def test_regular(self):
        self.assertEqual(TreeTokenizer.deep_detokenize([
            [[31, 2], [60, 57, 59, 53, 2], [50, 57, 55, 2], [50, 69, 68, 68, 67, 9, 2]],
            [[31, 2], [51, 49, 62, 2], [62, 63, 68, 2], [60, 57, 53, 9, 2]]
        ], level=2), self.dummy_expected[0])

        self.assertEqual(TreeTokenizer.deep_detokenize([
            [[67, 63, 61, 53, 2], [63, 68, 56, 53, 66, 2], [67, 63, 62, 55, 9, 2]]
        ], level=2), self.dummy_expected[1])

    def test_no_eos(self):
        self.assertEqual(TreeTokenizer.deep_detokenize([
            [[31, 2], [60, 57, 59, 53, 2], [50, 57, 55, 2], [50, 69, 68, 68, 67, 9], [2]],
            [[31, 2], [51, 49, 62, 2], [62, 63, 68, 2], [60, 57, 53, 9, 2]]
        ], level=2), self.dummy_expected[0])

    def test_join(self):
        self.assertEqual(TreeTokenizer.deep_detokenize([
            [[31, 2], [60, 57, 59, 53, 2], [50, 57, 55, 2], [50, 69, 68, 68, 67, 9, 2], -1,
             [31, 2], [51, 49, 62, 2], [62, 63, 68, 2], [60, 57, 53, 9, 2]]
        ], level=2), self.dummy_expected[0])

    def test_join_no_eos(self):
        self.assertEqual(TreeTokenizer.deep_detokenize([
            [[31, 2], [60, 57, 59, 53, 2], [50, 57, 55, 2], [50, 69, 68, 68, 67, 9], [2], -1,
             [31, 2], [51, 49, 62, 2], [62, 63, 68, 2], [60, 57, 53, 9, 2]]
        ], level=2), self.dummy_expected[0])

    def test_all_eos(self):
        self.assertEqual(TreeTokenizer.deep_detokenize([
            [-1, -1, -1, -1, -1, -1], [-1, -1, -1, -1, -1, -1]
        ], level=2), '<s>')

    def test_join_sentences(self):
        self.assertEqual(TreeTokenizer.deep_detokenize([
            [[67, 63, 61, 53, 2], [63, 68, 56, 53, 66, 2], [67, 63, 62, 55, 9, 2]],
            -1,
            [[67, 63, 61, 53, 2], [63, 68, 56, 53, 66, 2], [67, 63, 62, 55, 9, 2]]
        ], level=2), 'some other song.<p>some other song.')
