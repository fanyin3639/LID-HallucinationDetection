import re
import string
import nltk
import evaluate
from sklearn import metrics


def normalize_text(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace.
    Copied from the [QuAC](http://quac.ai/) evaluation script found at
    https://s3.amazonaws.com/my89public/quac/scorer.py"""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))


def f1_score(preds, golds):
    scores = []

    for pred in preds:
        ps = 0.0
        for gold in golds:
            ret = nltk.f_measure(set(normalize_text(pred).split()), set(normalize_text(gold).split()))
            if ret is None:
                ret = 0.0
            if ret > ps:
                ps = ret
        scores.append(ps)
    return max(scores)

rouge = evaluate.load('rouge')
def rouge_L(preds, golds):
    scores = []
    for pred in preds:
        ps = 0.0
        for gold in golds:
            pred = normalize_text(pred)
            gold = normalize_text(gold)
            rouge_scores = rouge.compute(predictions=[pred], references=[gold])['rougeL']
            if rouge_scores > ps:
                ps = rouge_scores
        scores.append(ps)
    return max(scores)


def roc(labels, scores):
    auroc = metrics.roc_auc_score(labels, scores)
    return auroc
