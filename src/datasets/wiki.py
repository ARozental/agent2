import re

from src.pre_processing import Splitters, TreeTokenizer
from src.pre_processing.dataset import Dataset
from src.config import Config
from datasets import load_dataset
import os

ARTICLE_HEADER = re.compile(r'_START_ARTICLE_\n.*\n')
SECTION_HEADER = re.compile(r'_START_SECTION_\n.*\n')
PARAGRAPH_REGEX = re.compile(r'_START_PARAGRAPH_\n')


def article_to_sections(text):
    return SECTION_HEADER.split(text)


def section_to_paragraphs(text):
    return PARAGRAPH_REGEX.split(text)


class WikiDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        tmp_folder = os.path.join(dir_path, '..', '..', 'tmp', 'huggingface')
        self.dataset = load_dataset('wiki40b', 'en', cache_dir=tmp_folder)

    def init_tree_tokenizer(self):
        TreeTokenizer.split_functions = [
            Splitters.sentence_to_words,
            Splitters.paragraph_to_sentences,
            section_to_paragraphs,
            article_to_sections,
        ]

    @staticmethod
    def _parse_article(article):
        text = article['text']

        # Strip article title
        text = ARTICLE_HEADER.sub('', text)

        # Remove section titles if not learning them (section == chapter)
        if Config.agent_level < Config.levels['CHAPTER']:
            text = SECTION_HEADER.sub('', text)

        # Remove paragraphs if not learning them
        if Config.agent_level < Config.levels['PARAGRAPH']:
            text = PARAGRAPH_REGEX.sub('', text)

        return text

    def __getitem__(self, index):
        article = self.dataset['train'][index]
        return self._parse_article(article)

    def __iter__(self):
        for article in self.dataset['train']:
            yield self._parse_article(article)
