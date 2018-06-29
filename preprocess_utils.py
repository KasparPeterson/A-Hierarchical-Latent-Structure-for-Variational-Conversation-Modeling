from model.utils import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
import random


def pad_sentences(conversations, max_sentence_length=30, max_conversation_length=10):
    def pad_tokens(tokens, max_sentence_length=max_sentence_length):
        n_valid_tokens = len(tokens)
        if n_valid_tokens > max_sentence_length - 1:
            tokens = tokens[:max_sentence_length - 1]
        n_pad = max_sentence_length - n_valid_tokens - 1
        tokens = tokens + [EOS_TOKEN] + [PAD_TOKEN] * n_pad
        return tokens

    def pad_conversation(conversation):
        conversation = [pad_tokens(sentence) for sentence in conversation]
        return conversation

    all_padded_sentences = []
    all_sentence_length = []

    for conversation in conversations:
        if len(conversation) > max_conversation_length:
            conversation = conversation[:max_conversation_length]
        sentence_length = [min(len(sentence) + 1, max_sentence_length)  # +1 for EOS token
                           for sentence in conversation]
        all_sentence_length.append(sentence_length)

        sentences = pad_conversation(conversation)
        all_padded_sentences.append(sentences)

    # [n_conversations, n_sentence (various), max_sentence_length]
    sentences = all_padded_sentences
    # [n_conversations, n_sentence (various)]
    sentence_length = all_sentence_length
    return sentences, sentence_length


def train_valid_test_split_by_conversation(conversations, split_ratio=[0.8, 0.1, 0.1]):
    """Train/Validation/Test split by randomly selected movies"""

    train_ratio, valid_ratio, test_ratio = split_ratio
    assert train_ratio + valid_ratio + test_ratio == 1.0

    n_conversations = len(conversations)

    # Random shuffle movie list
    random.seed(0)
    random.shuffle(conversations)

    # Train / Validation / Test Split
    train_split = int(n_conversations * train_ratio)
    valid_split = int(n_conversations * (train_ratio + valid_ratio))

    train = conversations[:train_split]
    valid = conversations[train_split:valid_split]
    test = conversations[valid_split:]

    print('Train set:', len(train), 'conversations')
    print('Validation set:', len(valid), 'conversations')
    print('Test set:', len(test), 'conversations')

    return train, valid, test
