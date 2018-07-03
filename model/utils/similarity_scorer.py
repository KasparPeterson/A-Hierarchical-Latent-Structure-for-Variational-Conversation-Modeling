import nltk
import difflib
from nltk.translate.bleu_score import SmoothingFunction

smoothie = SmoothingFunction().method4


# The higher the better
def get_bleu_score(hypothesis, reference):
    try:
        return nltk.translate.bleu_score.sentence_bleu(
            [reference.split()],
            hypothesis.split(),
            smoothing_function=smoothie)
    except:
        return 0


def get_sequence_matcher_score(hypothesis, reference):
    return difflib.SequenceMatcher(None, hypothesis, reference).ratio()


def get_levenshtein_score(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for index2, char2 in enumerate(s2):
        newDistances = [index2 + 1]
        for index1, char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1 + 1],
                                             newDistances[-1])))
        distances = newDistances
    return distances[-1]


if __name__ == '__main__':
    hypothesis = 'It is a cat at the room'
    reference = 'It is a cat inside the room'

    print("Bleu:", get_bleu_score(hypothesis, reference))
    print("Secquence:", get_sequence_matcher_score(hypothesis, reference))
    print("Levenshtein:", get_levenshtein_score(hypothesis, reference))
