# Version 1.1
# 11/20/2022

import random
from collections import defaultdict, Counter
from operator import itemgetter
from typing import Iterable, Generator, Sequence

# DO NOT MODIFY
RANDOM_SEED = 12345
random.seed(RANDOM_SEED)


# DO NOT MODIFY
class ClassificationInstance:
    """Represent a label and features for classification."""

    def __init__(self, label: str, features: Iterable[str]) -> None:
        self.label: str = label
        # Features can be passed in as any iterable and they will be
        # stored in a tuple
        self.features: tuple[str, ...] = tuple(features)

    def __repr__(self) -> str:
        return f"<ClassificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.label}; features={self.features}"


# DO NOT MODIFY
class LanguageIdentificationInstance:
    """Represent a single instance from a language ID dataset."""

    def __init__(
            self,
            language: str,
            text: str,
    ) -> None:
        self.language: str = language
        self.text: str = text

    def __repr__(self) -> str:
        return f"<LanguageIdentificationInstance: {str(self)}>"

    def __str__(self) -> str:
        return f"label={self.language}; text={self.text}"

    # You should never call this function directly. It's called by data loading functions.
    @classmethod
    def from_line(cls, line: str) -> "LanguageIdentificationInstance":
        splits = line.strip().split("\t")
        assert len(splits) == 2
        return cls(splits[0], splits[1])


# DO NOT MODIFY
def load_lid_instances(
        path: str,
) -> Generator[LanguageIdentificationInstance, None, None]:
    """Load airline sentiment instances from a JSON file."""
    with open(path, encoding="utf8") as file:
        for line in file:
            yield LanguageIdentificationInstance.from_line(line)


# DO NOT MODIFY
def max_item(scores: dict[str, float]) -> tuple[str, float]:
    """Return the key and value with the highest value."""
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return max(scores.items(), key=itemgetter(1))


# DO NOT MODIFY
def items_descending_value(counts: Counter[str]) -> list[str]:
    """Return the keys in descending frequency, breaking ties lexicographically."""
    # Why can't we just use most_common? It sorts by descending frequency, but items
    # of the same frequency follow insertion order, which we can't depend on.
    # Why can't we just use sorted with reverse=True? It will give us descending
    # by count, but reverse lexicographic sorting, which is confusing.
    # So instead we used sorted() normally, but for the key provide a tuple of
    # the negative value and the key.
    # PyCharm gives a false positive type warning here
    # noinspection PyTypeChecker
    return [key for key, value in sorted(counts.items(), key=_items_sort_key)]


# DO NOT MODIFY
def _items_sort_key(item: tuple[str, int]) -> tuple[int, str]:
    # This is used by items_descending_count, but you should never call it directly.
    return -item[1], item[0]


############################################################
# The stubs below this are the ones that you should fill in.
# Do not modify anything above this line other than to add any needed imports.


class CharBigramFeatureExtractor:
    @staticmethod
    def extract_features(
            instance: LanguageIdentificationInstance,
    ) -> ClassificationInstance:
        """Extract character bigram features from an instance."""
        label = instance.language
        features = set()
        for ind in range(len(instance.text) - 1):
            features.add(instance.text[ind:ind + 2])
        return ClassificationInstance(label, features)


class InstanceCounter:
    def __init__(self) -> None:
        self.label_counter = Counter()
        self.sorted_labels = []

    def count_instances(self, instances: Iterable[ClassificationInstance]) -> None:
        """Count the labels in the provided instances."""
        for instance in instances:
            self.label_counter[instance.label] += 1
        self.sorted_labels = items_descending_value(self.label_counter)

    def labels(self) -> list[str]:
        """Return a sorted list of the labels."""
        return self.sorted_labels


class Perceptron:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels
        self.weights = dict(defaultdict(float))
        self.sums = dict(defaultdict(float))
        self.last_updated = dict(defaultdict(int))
        self.prediction_list = list()
        for label in self.labels:
            self.weights[label] = defaultdict(float)
            self.sums[label] = defaultdict(float)
            self.last_updated[label] = defaultdict(int)

    def classify(self, features: Iterable[str]) -> str:
        scores = dict()
        for label in self.labels:
            scores[label] = 0
            for feature in features:
                scores[label] += self.weights[label][feature]
        return max_item(scores)[0]

    def learn(
            self,
            instance: ClassificationInstance,
            step: int,
            lr: float,
    ) -> None:
        prediction = self.classify(instance.features)
        if prediction == instance.label:
            return
        for feature in instance.features:
            self.sums[instance.label][feature] += (step - self.last_updated[instance.label][feature]) * (self.weights[instance.label][feature])
            self.sums[prediction][feature] += (step - self.last_updated[prediction][feature]) * (self.weights[prediction][feature])
            self.weights[instance.label][feature] += lr
            self.weights[prediction][feature] -= lr
            self.last_updated[instance.label][feature] = step
            self.last_updated[prediction][feature] = step

    def predict(self, test: Sequence[ClassificationInstance]) -> list[str]:
        for instance in test:
            self.prediction_list.append(self.classify(instance.features))
        return self.prediction_list

    def average(self, final_step: int) -> None:
        for label in self.weights:
            for feature in self.weights[label]:
                self.sums[label][feature] += (final_step - self.last_updated[label][feature]) * self.weights[label][feature]
                self.weights[label][feature] = self.sums[label][feature] / final_step


def train_perceptron(
        model: Perceptron,
        data: list[ClassificationInstance],
        epochs: int,
        lr: float,
        *,
        average: bool,
) -> None:
    # DO NOT MODIFY THE ASSERT STATEMENTS
    # Some argument checks to avoid any accidents
    assert isinstance(model, Perceptron)
    assert isinstance(data, list)
    assert data
    assert isinstance(data[0], ClassificationInstance)
    assert isinstance(epochs, int)
    assert epochs > 0
    assert isinstance(lr, float)
    assert lr > 0
    assert isinstance(average, bool)

    # Add your code here
    step = 1
    for epoch_count in range(epochs):
        for instance in data:
            model.learn(instance, step, lr)
            step += 1
        random.shuffle(data)
    if average:
        model.average(step)
