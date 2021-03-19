from collections import defaultdict
import glob
import re


def is_empty(a):
    if a == [0]:
        return True
    if type(a) != list:
        return False
    elif len(a) > 1:
        return False
    elif len(a) == 0:
        return True
    elif type(a[0]) != list:
        return False
    else:
        return is_empty(a[0])


# letters,words,sentences,paragraphs,chapters (though it is "Chapter " really)
delimiters = ['', ' ', '  ', '\n', '\n\n']


def split_text(text, tokenize_fn, current_level, config):
    ll = config.sequence_lengths[current_level] - 1
    if current_level == 0:
        return text[0:ll]
    f = lambda text: text.split(delimiters[current_level])
    return [split_text(t, tokenize_fn, current_level - 1, config) for t in f(text)][0:ll]


def create_tokenizer(char_file):  # "../chars.txt"
    x = open(char_file, "r", encoding='utf-8').readlines()
    tokenizer = defaultdict(int, dict(zip([l.strip() for l in x], range(1, 7777))))

    def tokenize(word):
        return [tokenizer[l] for l in word]

    return tokenize


def remove_empty(a):
    if type(a) != list:
        return a
    return [remove_empty(x) for x in a if is_empty(x) == False]


def do_book(location, config):
    tokenize_fn = create_tokenizer("chars.txt")
    lines = open(location, "r", encoding='utf-8').readlines()
    # book_title = lines[10].strip()
    text = "".join(lines)
    print(text)
    print('---')
    chapters = re.split('(?i)chapter ', text)
    print(chapters)
    print('---')
    ll = config.sequence_lengths[4] - 1
    chapters = [split_text("Chapter " + x, tokenize_fn, 3, config) for x in chapters][1:ll + 1]
    print(chapters)
    # chapters[0] = split_text(book_title,tokenize_fn, 3,config)
    return remove_empty(chapters)


class BookDataset(object):
    def __init__(self, config):
        self.config = config
        self.iter = self.build_iterator()

    def get_next_book(self):
        all_books = glob.glob("datasets/no_titles/*.txt")
        l = len(all_books)
        num = 0
        while True:
            out = do_book(all_books[num % l], self.config)
            ragged_out = tf.ragged.constant(out)
            yield ({
                "flat_values": ragged_out.flat_values,
                "nested_row_splits": ragged_out.nested_row_splits
            })
            num += 1

    def build_iterator(self):
        batch_size = self.config.batch_size
        prefetch_batch_buffer = 2
        dataset = tf.data.Dataset.from_generator(self.get_next_book, output_types={
            "flat_values": tf.int64, "nested_row_splits": (tf.int64, tf.int64, tf.int64, tf.int64)
        })

        dataset = dataset.map(
            lambda h: {"tokens": tf.RaggedTensor.from_nested_row_splits(h["flat_values"], (h["nested_row_splits"]))})

        return it
