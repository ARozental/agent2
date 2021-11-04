from src.config import Config
from src.pre_processing import Splitters, TreeTokenizer
from src.pre_processing.dataset import Dataset
from wikiextractor import WikiExtractor
from xml.etree import ElementTree
import urllib.request
import shutil
import glob
import json
import bz2
import sys
import os
import re

SECTION_HEADER = re.compile(r'_START_SECTION_\n.*\n')
PARAGRAPH_REGEX = re.compile(r'\n')


def article_to_sections(text):
    return SECTION_HEADER.split(text)


def section_to_paragraphs(text):
    return PARAGRAPH_REGEX.split(text)


class SimpleWikiDataset(Dataset):
    """
    Article = Book
    Section = Chapter
    """

    def __init__(self, max_num=None, **kwargs):
        super().__init__(**kwargs)

        if Config.storage_location is not None:
            raise NotImplementedError('SimpleWiki is not implemented for GCS storage yet.')

        dir_path = os.path.dirname(os.path.realpath(__file__))
        tmp_folder = os.path.join(dir_path, '..', '..', 'tmp', 'simple_wiki')
        data_file = os.path.join(tmp_folder, 'data.json')
        if not os.path.exists(data_file):
            self.download_extract(tmp_folder)

        if os.path.exists(data_file):
            with open(data_file) as f:
                self.dataset = [json.loads(line)['text'] for line in f]

        self.max_num = max_num
        if self.max_num is None:
            self.data = list(range(len(self.dataset)))
        else:
            self.data = list(range(self.max_num))

        # TPU bootup takes way too long for now with this on
        # if Config.use_tpu:
        #     assert Config.dynamic_node_sizes is True

    @staticmethod
    def download_extract(output_folder):
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        def parse_article(article):
            raw_text = article.find('ns:revision/ns:text', namespaces=namespaces).text
            if raw_text is None:
                titles = []
            else:
                titles = [title[1] for title in section.findall(raw_text)]

            return {
                'id': article.find('ns:id', namespaces=namespaces).text,
                'sections': titles,
            }

        print('Downloading SimpleWiki Dump...')
        url = 'https://dumps.wikimedia.org/simplewiki/20211101/simplewiki-20211101-pages-articles.xml.bz2'
        urllib.request.urlretrieve(url, os.path.join(output_folder, 'raw.xml.bz2'))

        print('Decompressing dump')
        with bz2.BZ2File(os.path.join(output_folder, 'raw.xml.bz2')) as f:
            data = f.read()

        with open(os.path.join(output_folder, 'raw.xml'), 'wb') as f:
            f.write(data)

        xml_file = os.path.join('tmp', 'simple_wiki', 'raw.xml')

        namespaces = {'ns': 'http://www.mediawiki.org/xml/export-0.10/'}
        section = re.compile(r'(==+)\s*(.*?)\s*\1')

        print('Parsing section titles of articles')
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        articles = [parse_article(article) for article in root.findall('ns:page', namespaces=namespaces)]
        article_mapping = {article['id']: article['sections'] for article in articles}

        print('Parsing article content')
        old_argv = sys.argv
        sys.argv = ['', '-o', os.path.join(output_folder, 'extract'), xml_file, '--json']
        WikiExtractor.main()
        sys.argv = old_argv

        print('Creating final dataset file')
        with open(os.path.join(output_folder, 'data.json'), 'w') as w:
            for filename in glob.iglob(os.path.join(output_folder, 'extract', '**', '**'), recursive=True):
                if os.path.isdir(filename):
                    continue

                with open(filename) as f:
                    for line in f:
                        article = json.loads(line)
                        if article['text'].strip() == '':
                            continue

                        del article['url']
                        sections = article_mapping[article['id']]
                        for section in sections:
                            search_text = '\n' + section + '.\n'
                            if search_text in article['text']:
                                article['text'] = article['text'].replace(search_text,
                                                                          '\n_START_SECTION_\n' + section + '\n')
                        article['text'] = article['text'].strip()
                        w.write(json.dumps(article) + '\n')

        # Cleanup
        shutil.rmtree(os.path.join(output_folder, 'extract'))
        os.remove(os.path.join(output_folder, 'raw.xml.bz2'))
        os.remove(os.path.join(output_folder, 'raw.xml'))

        # Download the .xml.bz2 file
        pass

        # Extract the .xml file
        pass

        # Parse it and get all of the section names for later
        pass

        # Run the WikiExtractor
        pass

        # Concatenate all of the .json files together
        pass

        # Update each json file so that the sections are identifiable
        pass

    def init_tree_tokenizer(self):
        TreeTokenizer.split_functions = [
            Splitters.sentence_to_words,
            Splitters.paragraph_to_sentences,
            section_to_paragraphs,
            article_to_sections,
        ]

    def valid_text(self, text):
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.dataset[index]

    @staticmethod
    def build_tensors(d1, d2):
        import torch
        tensors = {
            'lengths': {},
        }

        for parent_key in set(d1.keys()) | set(d2.keys()):  # All keys
            tensors[parent_key] = {}
            sub_keys = set(d1.get(parent_key, {}).keys()) | set(d2.get(parent_key, {}).keys())
            for key in sub_keys:
                if parent_key not in d1:
                    b = d2[parent_key][key]
                    a = torch.zeros(b.shape, dtype=b.dtype)
                    tensors['lengths'][parent_key + '-' + key] = torch.tensor([0, b.shape[0]])
                elif parent_key not in d2:
                    a = d1[parent_key][key]
                    b = torch.zeros(a.shape, dtype=a.dtype)
                    tensors['lengths'][parent_key + '-' + key] = torch.tensor([a.shape[0], 0])
                else:
                    a = d1[parent_key][key]
                    b = d2[parent_key][key]
                    tensors['lengths'][parent_key + '-' + key] = torch.tensor([a.shape[0], b.shape[0]])
                    if a.shape[0] < b.shape[0]:
                        new_shape = list(a.shape)
                        new_shape[0] = b.shape[0] - a.shape[0]
                        a = torch.cat([a, torch.zeros(new_shape, dtype=a.dtype)], dim=0)
                    else:
                        new_shape = list(b.shape)
                        new_shape[0] = a.shape[0] - b.shape[0]
                        b = torch.cat([b, torch.zeros(new_shape, dtype=a.dtype)], dim=0)
                tensors[parent_key][key] = torch.stack([a, b])

        return tensors

    def __iter__(self):
        last_articles = None
        for batch in range(len(self.data) // Config.batch_size):
            begin = batch * Config.batch_size
            articles = [self.dataset[begin + num] for num in range(Config.batch_size)]

            if Config.multi_gpu:
                if batch % 2 == 0:
                    last_articles = articles
                    continue

                data1 = TreeTokenizer.batch_texts_to_trees(last_articles)
                data2 = TreeTokenizer.batch_texts_to_trees(articles)

                batch_roots = [data1[0], data2[0]]
                tensors = self.build_tensors(data1[1], data2[1])

                yield batch_roots, tensors
            else:
                yield TreeTokenizer.batch_texts_to_trees(articles)
