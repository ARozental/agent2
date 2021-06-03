from src.pre_processing import Splitters, TreeTokenizer
from src.pre_processing.local_dataset import LocalDataset
from src.config import Config
import os
import re
PARAGRAPH_REGEX = re.compile(r'\n+')


def chapter_to_paragraphs(text):
    return PARAGRAPH_REGEX.split(text)


class TestExample(LocalDataset):
    def __init__(self, text,level=1,**kwargs):
        with open("datasets/test_example/example1.txt", "w") as file1:
        # Writing data to a file
          file1.write(text)

        super().__init__(folder=os.path.join('datasets', 'test_example', '*.txt'), **kwargs)
        assert Config.agent_level <= Config.levels['PARAGRAPH']

    def init_tree_tokenizer(self):
        TreeTokenizer.split_functions = [
            Splitters.sentence_to_words,
            Splitters.paragraph_to_sentences,
            chapter_to_paragraphs
        ]
