import pickle
import preprocess_utils
import os
from pathlib import Path
from multiprocessing import Pool
from model.utils import Tokenizer, Vocab
from tqdm import tqdm

project_dir = Path(__file__).resolve().parent
datasets_dir = project_dir.joinpath('datasets/')
cornell_dir = datasets_dir.joinpath('trump/')

INPUT_FILE = "datasets/trump/trump_dirty.txt"
OUTPUT_FILE = "datasets/trump/trump_cleaned.txt"
GROUP_BY = 3

tokenizer = Tokenizer('spacy')


def get_last_line(conversation):
    if len(conversation) > 0:
        return conversation[len(conversation) - 1]
    return ""


def get_conversations():
    result = []
    conversation = []
    last = ""
    with open(INPUT_FILE, "r") as f:
        for line in f:
            line = line.replace("\n", "")
            if len(line) > 0 and line != get_last_line(conversation):
                conversation.append(line)
            if len(conversation) == GROUP_BY:
                result.append(conversation)
                conversation = []

            if line == last:
                print(line)
            last = line
    return result


def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def tokenize_conversation(lines):
    sentence_list = [tokenizer(line) for line in lines]
    return sentence_list


if __name__ == '__main__':
    max_sent_len = 30
    max_vocab_size = 20000
    max_conv_len = 10
    min_freq = 5
    n_workers = os.cpu_count()

    conversations = get_conversations()
    with open(OUTPUT_FILE, "w") as f:
        for c in conversations:
            f.write("%s\n" % c)

    train, valid, test = preprocess_utils.train_valid_test_split_by_conversation(conversations)

    for split_type, conv_objects in [('train', train), ('valid', valid), ('test', test)]:
        print('Processing', split_type, 'dataset...')
        split_data_dir = cornell_dir.joinpath(split_type)
        split_data_dir.mkdir(exist_ok=True)

        print('Tokenize.. (n_workers=', n_workers, ')')


        def _tokenize_conversation(conv):
            return tokenize_conversation(conv)


        with Pool(n_workers) as pool:
            conversations = list(tqdm(pool.imap(_tokenize_conversation, conv_objects),
                                      total=len(conv_objects)))

        conversation_length = [min(len(conv), max_conv_len)
                               for conv in conv_objects]

        sentences, sentence_length = preprocess_utils.pad_sentences(
            conversations,
            max_sentence_length=max_sent_len,
            max_conversation_length=max_conv_len)

        print('Saving preprocessed data at', split_data_dir)
        to_pickle(conversation_length, split_data_dir.joinpath('conversation_length.pkl'))
        to_pickle(sentences, split_data_dir.joinpath('sentences.pkl'))
        to_pickle(sentence_length, split_data_dir.joinpath('sentence_length.pkl'))

        if split_type == 'train':
            print('Save Vocabulary...')
            vocab = Vocab(tokenizer)
            vocab.add_dataframe(conversations)
            vocab.update(max_size=max_vocab_size, min_freq=min_freq)

            print('Vocabulary size: ', len(vocab))
            vocab.pickle(cornell_dir.joinpath('word2id.pkl'), cornell_dir.joinpath('id2word.pkl'))

    print('Done!')
