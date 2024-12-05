import nltk
import random
from nltk.corpus import wordnet
from typing import List, Optional

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

class TextAugmenter:
    def __init__(self, synonym_prob: float = 0.1, deletion_prob: float = 0.1, max_aug_per_sample: int = 2):
        self.synonym_prob = synonym_prob
        self.deletion_prob = deletion_prob
        self.max_aug_per_sample = max_aug_per_sample

    def get_synonyms(self, word: str) -> List[str]:
        synonyms = []
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                if lemma.name() != word:
                    synonyms.append(lemma.name())
        return list(set(synonyms))

    def synonym_replacement(self, text: str) -> str:
        words = text.split()
        new_words = words.copy()
        
        for idx, word in enumerate(words):
            if random.random() < self.synonym_prob:
                synonyms = self.get_synonyms(word)
                if synonyms:
                    new_words[idx] = random.choice(synonyms)
        
        return ' '.join(new_words)

    def random_deletion(self, text: str) -> str:
        words = text.split()
        if len(words) == 1:
            return text

        new_words = []
        for word in words:
            if random.random() > self.deletion_prob:
                new_words.append(word)

        if not new_words:
            new_words = [random.choice(words)]

        return ' '.join(new_words)

    def augment(self, text: str) -> List[str]:
        augmented_texts = []
        num_aug = random.randint(1, self.max_aug_per_sample)
        
        for _ in range(num_aug):
            aug_text = text
            if random.random() < 0.5:
                aug_text = self.synonym_replacement(aug_text)
            else:
                aug_text = self.random_deletion(aug_text)
            augmented_texts.append(aug_text)
        
        return augmented_texts
