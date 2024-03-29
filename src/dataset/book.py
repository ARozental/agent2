from src.pre_processing import Splitters, TreeTokenizer
from src.pre_processing.local_dataset import LocalDataset
import os
import re

CHAPTER_REGEX = re.compile(r'Chapter .*\n')
PARAGRAPH_REGEX = re.compile(r'\n+')


def chapter_to_paragraphs(text):
    return PARAGRAPH_REGEX.split(text)


def book_to_chapters(text):
    return [chapter for chapter in CHAPTER_REGEX.split(text) if len(chapter) > 0]


class BookDataset(LocalDataset):
    # sequence lengths for all five books: [47, 72, 23, 200, 21]
    def __init__(self, no_stats=False, **kwargs):
        if no_stats:
            folder = os.path.join('datasets', 'no_stats_books', '*.txt')
        else:
            raise NotImplementedError
        super().__init__(folder, **kwargs)

    def init_tree_tokenizer(self):
        TreeTokenizer.split_functions = [
            Splitters.sentence_to_words,
            Splitters.paragraph_to_sentences,
            chapter_to_paragraphs,
            book_to_chapters,
        ]
