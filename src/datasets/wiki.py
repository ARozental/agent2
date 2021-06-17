from src.config import Config
from src.pre_processing import Splitters, TreeTokenizer
from src.pre_processing.dataset import Dataset
from datasets import load_dataset, load_from_disk
import os
import re

ARTICLE_HEADER = re.compile(r'_START_ARTICLE_\n.*\n')
SECTION_HEADER = re.compile(r'_START_SECTION_\n.*\n')
PARAGRAPH_REGEX = re.compile(r'(?:_START_PARAGRAPH_\n?| ?_NEWLINE_)')


def article_to_sections(text):
    return SECTION_HEADER.split(text)


def section_to_paragraphs(text):
    return PARAGRAPH_REGEX.split(text)


class WikiDataset(Dataset):
    """
    Article = Book
    Section = Chapter
    """

    def __init__(self, max_num=None, **kwargs):
        super().__init__(**kwargs)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        tmp_folder = os.path.join(dir_path, '..', '..', 'tmp', 'huggingface')

        if os.path.exists(tmp_folder) or Config.storage_location is None:
            self.dataset = load_dataset('wiki40b', 'en', cache_dir=tmp_folder)
        elif Config.storage_location is not None:
            import gcsfs
            gcs = gcsfs.GCSFileSystem()
            location = Config.storage_location.replace('gs://', 'gcs://')
            self.dataset = load_from_disk(os.path.join(location, 'tmp', 'huggingface', 'wiki40b/en/1.1.0/'), fs=gcs)

        self.max_num = max_num
        if self.max_num is None:
            self.data = list(range(len(self.dataset['train'])))
        else:
            self.data = list(range(self.max_num))

        # TPU bootup takes way too long for now with this on
        # if Config.use_tpu:
        #     assert Config.dynamic_node_sizes is True

    def init_tree_tokenizer(self):
        TreeTokenizer.split_functions = [
            Splitters.sentence_to_words,
            Splitters.paragraph_to_sentences,
            section_to_paragraphs,
            article_to_sections,
        ]

    def valid_text(self,text):
      return True

    @staticmethod
    def _parse_article(article):
        return ARTICLE_HEADER.sub('', article['text']).strip()  # Strip article title

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        article = self.dataset['train'][index]
        return self._parse_article(article)

    def __iter__(self):
        for batch in range(len(self.data) // Config.batch_size):
            begin = batch * Config.batch_size
            articles = [self._parse_article(self.dataset['train'][begin + num]) for num in range(Config.batch_size)]
            yield TreeTokenizer.batch_texts_to_trees(articles)
