# Version 1.1
# 11/20/2022

import os
import random
import unittest
from collections import defaultdict
from typing import Any

from sklearn.metrics import accuracy_score

from grader import Grader, points, timeout
from hw5 import (
    Perceptron,
    ClassificationInstance,
    InstanceCounter,
    CharBigramFeatureExtractor,
    LanguageIdentificationInstance,
    load_lid_instances,
    train_perceptron,
    RANDOM_SEED,
)

DEV_DATA = os.path.join("test_data", "mot_dev.tsv")
TRAIN_DATA = os.path.join("test_data", "mot_train.tsv")
# We store this is a tuple to prevent accidental modification, but
# it needs to be converted to a list before training.
SENTIMENT_DATA = (
    ClassificationInstance(
        "positive",
        ["I", "love", "tacos", "!"],
    ),
    ClassificationInstance(
        "negative",
        ["I", "dislike", "broccoli", "."],
    ),
    ClassificationInstance(
        "negative",
        [
            "I",
            "love",
            "to",
            "dislike",
            "tacos",
        ],
    ),
)


class TestFeatureExtraction(unittest.TestCase):
    @points(4)
    def test_bigram_features(self) -> None:
        """Bigram features are correctly generated."""
        example_sentence = LanguageIdentificationInstance("eng", "hello!")
        features = set(
            CharBigramFeatureExtractor().extract_features(example_sentence).features
        )
        self.assertSetEqual(
            {"lo", "el", "he", "o!", "ll"},
            features,
        )

    @points(1)
    def test_feature_case(self) -> None:
        """Casing is preserved."""
        example_sentence = LanguageIdentificationInstance("eng", "HeLlO!")
        features = set(
            CharBigramFeatureExtractor().extract_features(example_sentence).features
        )
        self.assertSetEqual(
            {"lO", "eL", "O!", "Ll", "He"},
            features,
        )


class TestInstanceCounter(unittest.TestCase):
    @points(1)
    def test_label_order(self) -> None:
        """Labels are sorted correctly."""
        labels = ["a", "c", "c", "b", "b"]
        instances = [ClassificationInstance(label, ("feature")) for label in labels]
        counter = InstanceCounter()
        counter.count_instances(instances)
        self.assertListEqual(["b", "c", "a"], counter.labels())


class TestNoAveraging(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(RANDOM_SEED)
        instance_counter = InstanceCounter()
        instance_counter.count_instances(SENTIMENT_DATA)
        self.labels = instance_counter.labels()

    @points(1)
    def test_labels(self) -> None:
        """Labels are stored correctly in the model."""
        model = Perceptron(self.labels)
        self.assertEqual(self.labels, model.labels)

    @points(1)
    def test_weight_dicts(self) -> None:
        """Weights are stored correctly in the model."""
        model = Perceptron(self.labels)
        # Must be a dictionary, not a defaultdict
        self.assertEqual(dict, type(model.weights))
        for label in self.labels:
            # Must be a defaultdict
            self.assertEqual(defaultdict, type(model.weights[label]))

    @points(4)
    def test_default_prediction(self) -> None:
        """The default prediction is the first label in the list of labels."""
        model = Perceptron(self.labels)
        self.assertEqual("negative", model.classify(SENTIMENT_DATA[0].features))

    @points(8)
    def test_first_update(self) -> None:
        """Weights are correct after the first training example."""
        # Test for two different LR values
        for lr in (1.0, 0.5):
            model = Perceptron(self.labels)
            data = [SENTIMENT_DATA[0]]
            # Don't need to worry about random seed here since there's only one instance
            train_perceptron(model, data, 1, lr, average=False)
            self.assertEqual(lr, model.weights["positive"]["I"])
            self.assertEqual(lr, model.weights["positive"]["love"])
            self.assertEqual(lr, model.weights["positive"]["tacos"])
            self.assertEqual(lr, model.weights["positive"]["!"])
            self.assertEqual(-lr, model.weights["negative"]["I"])
            self.assertEqual(-lr, model.weights["negative"]["love"])
            self.assertEqual(-lr, model.weights["negative"]["tacos"])
            self.assertEqual(-lr, model.weights["negative"]["!"])

    @points(8)
    def test_second_update(self) -> None:
        """Weights are correct after the second training example."""
        model = Perceptron(self.labels)
        data = list(SENTIMENT_DATA[:2])
        lr = 1.0
        train_perceptron(model, data, 1, lr, average=False)

        self.assertEqual(lr, model.weights["positive"]["love"])
        self.assertEqual(lr, model.weights["positive"]["tacos"])
        self.assertEqual(lr, model.weights["positive"]["!"])
        self.assertEqual(-lr, model.weights["negative"]["love"])
        self.assertEqual(-lr, model.weights["negative"]["tacos"])
        self.assertEqual(-lr, model.weights["negative"]["!"])

        self.assertEqual(-lr, model.weights["positive"]["dislike"])
        self.assertEqual(-lr, model.weights["positive"]["broccoli"])
        self.assertEqual(-lr, model.weights["positive"]["."])
        self.assertEqual(lr, model.weights["negative"]["dislike"])
        self.assertEqual(lr, model.weights["negative"]["broccoli"])
        self.assertEqual(lr, model.weights["negative"]["."])

        # Conflicting updates so the weight is zero
        for label in self.labels:
            self.assertEqual(0, model.weights[label]["I"])

    @points(8)
    def test_third_update(self) -> None:
        """Weights are correct after the third training example."""
        model = Perceptron(self.labels)
        data = list(SENTIMENT_DATA)
        train_perceptron(model, data, 1, 1.0, average=False)

        self.assertDictEqual(
            {
                "I": -1.0,
                "love": 0.0,
                "tacos": 0.0,
                "!": 1.0,
                "dislike": -2.0,
                "broccoli": -1.0,
                ".": -1.0,
                "to": -1.0,
            },
            model.weights["positive"],
        )
        self.assertDictEqual(
            {
                "I": 1.0,
                "love": 0.0,
                "tacos": 0.0,
                "!": -1.0,
                "dislike": 2.0,
                "broccoli": 1.0,
                ".": 1.0,
                "to": 1.0,
            },
            model.weights["negative"],
        )


class TestAveraging(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(RANDOM_SEED)
        instance_counter = InstanceCounter()
        instance_counter.count_instances(SENTIMENT_DATA)
        self.labels = instance_counter.labels()

    # Values are type annotated with Any because covariance is hard
    def assertWeightsEqual(
        self, expected: dict[str, Any], actual: dict[str, Any]
    ) -> None:
        # Get the union of keys and make sure they vale the same value
        all_keys = expected.keys() | actual.keys()
        for key in sorted(all_keys):
            self.assertAlmostEqual(
                expected[key],
                actual[key],
                msg=f"Values for key {repr(key)} differ",
            )

    @points(1)
    def test_averaged_empty(self) -> None:
        """Sums and last updated dicts are empty at the start."""
        model = Perceptron(self.labels)
        self.assertEqual(self.labels, list(model.sums))
        self.assertEqual(self.labels, list(model.last_updated))
        for label in self.labels:
            self.assertDictEqual({}, model.sums[label])
            self.assertDictEqual({}, model.last_updated[label])

    @points(1)
    def test_simple_averaged_weights(self) -> None:
        """Averaged weights are correct for a simple toy example."""
        data = [
            ClassificationInstance("positive", ["hooray"]),
            ClassificationInstance("positive", ["hooray"]),
        ]
        model = Perceptron(["negative", "positive"])
        train_perceptron(model, data, 1, 1.0, average=True)
        self.assertWeightsEqual(
            {"hooray": 2 / 3},
            model.weights["positive"],
        )
        self.assertWeightsEqual(
            {"hooray": -2 / 3},
            model.weights["negative"],
        )

    @points(6)
    def test_averaged_last_updated(self) -> None:
        """Last updated is correct after one epoch."""
        model = Perceptron(self.labels)
        train_perceptron(model, list(SENTIMENT_DATA), 1, 1.0, average=True)
        for label in model.labels:
            self.assertWeightsEqual(
                {
                    "I": 3,
                    "love": 3,
                    "tacos": 3,
                    "!": 1,
                    "dislike": 3,
                    "broccoli": 2,
                    ".": 2,
                    "to": 3,
                },
                model.last_updated[label],
            )

    @points(6)
    def test_averaged_sums(self) -> None:
        """Sums are correct after five epochs."""
        model = Perceptron(self.labels)
        train_perceptron(model, list(SENTIMENT_DATA), 5, 1.0, average=True)
        self.assertWeightsEqual(
            {
                "I": 1.0,
                "love": -13.0,
                "tacos": -13.0,
                "!": -26.0,
                "dislike": 27.0,
                "broccoli": 14.0,
                ".": 14.0,
                "to": 13.0,
            },
            model.sums["negative"],
        )
        self.assertWeightsEqual(
            {
                "I": -1.0,
                "love": 13.0,
                "tacos": 13.0,
                "!": 26.0,
                "dislike": -27.0,
                "broccoli": -14.0,
                ".": -14.0,
                "to": -13.0,
            },
            dict(model.sums["positive"]),
        )

    @points(7)
    def test_real_averaged_weights(self) -> None:
        """Averaged weights are correct after 1-3 epochs."""
        for num_epochs in (1, 2, 3):
            model = Perceptron(self.labels)
            random.seed(RANDOM_SEED)
            train_perceptron(model, list(SENTIMENT_DATA), num_epochs, 1.0, average=True)

            if num_epochs == 1:
                self.assertWeightsEqual(
                    {
                        "I": 0.0,
                        "love": -0.5,
                        "tacos": -0.5,
                        "!": -0.75,
                        "dislike": 0.75,
                        "broccoli": 0.5,
                        ".": 0.5,
                        "to": 0.25,
                    },
                    model.weights["negative"],
                )
                self.assertWeightsEqual(
                    {
                        "I": 0.0,
                        "love": 0.5,
                        "tacos": 0.5,
                        "!": 0.75,
                        "dislike": -0.75,
                        "broccoli": -0.5,
                        ".": -0.5,
                        "to": -0.25,
                    },
                    model.weights["positive"],
                )
            elif num_epochs == 2:
                self.assertWeightsEqual(
                    {
                        "I": 0.14285714285714285,
                        "love": -0.5714285714285714,
                        "tacos": -0.5714285714285714,
                        "!": -1.1428571428571428,
                        "dislike": 1.2857142857142858,
                        "broccoli": 0.7142857142857143,
                        ".": 0.7142857142857143,
                        "to": 0.5714285714285714,
                    },
                    model.weights["negative"],
                )
                self.assertWeightsEqual(
                    {
                        "I": -0.14285714285714285,
                        "love": 0.5714285714285714,
                        "tacos": 0.5714285714285714,
                        "!": 1.1428571428571428,
                        "dislike": -1.2857142857142858,
                        "broccoli": -0.7142857142857143,
                        ".": -0.7142857142857143,
                        "to": -0.5714285714285714,
                    },
                    model.weights["positive"],
                )
            elif num_epochs == 3:
                self.assertWeightsEqual(
                    {
                        "I": 0.1,
                        "love": -0.7,
                        "tacos": -0.7,
                        "!": -1.4,
                        "dislike": 1.5,
                        "broccoli": 0.8,
                        ".": 0.8,
                        "to": 0.7,
                    },
                    model.weights["negative"],
                )
                self.assertWeightsEqual(
                    {
                        "I": -0.1,
                        "love": 0.7,
                        "tacos": 0.7,
                        "!": 1.4,
                        "dislike": -1.5,
                        "broccoli": -0.8,
                        ".": -0.8,
                        "to": -0.7,
                    },
                    model.weights["positive"],
                )


class TestModelPredictions(unittest.TestCase):
    def setUp(self) -> None:
        feature_extractor = CharBigramFeatureExtractor()
        self.train_instances = [
            feature_extractor.extract_features(inst)
            for inst in load_lid_instances(TRAIN_DATA)
        ]
        self.dev_instances = [
            feature_extractor.extract_features(inst)
            for inst in load_lid_instances(DEV_DATA)
        ]
        self.dev_labels = [instance.label for instance in self.dev_instances]
        instance_counter = InstanceCounter()
        instance_counter.count_instances(self.train_instances)
        self.labels = instance_counter.labels()

    def train_eval_model(self, n_epochs: int, average: bool) -> float:
        model = Perceptron(self.labels)
        random.seed(RANDOM_SEED)
        train_perceptron(
            model,
            self.train_instances,
            n_epochs,
            1.0,
            average=average,
        )
        predictions = model.predict(self.dev_instances)
        return accuracy_score(self.dev_labels, predictions)

    @points(8)
    @timeout(16)
    def test_train_1_epoch(self) -> None:
        """Accuracy is high enough after one epoch."""
        accuracy = self.train_eval_model(1, False)
        # Solution gets .8848
        self.assertLessEqual(0.8846, accuracy)

    @points(8)
    @timeout(60)
    def test_train_5_epochs(self) -> None:
        """Accuracy is high enough after five epochs."""
        accuracy = self.train_eval_model(5, False)
        # Solution gets .9051
        self.assertLessEqual(0.9049, accuracy)

    @points(8)
    @timeout(60)
    def test_train_5_epochs_averaging(self) -> None:
        """Accuracy is high enough after five epochs with averaging."""
        accuracy = self.train_eval_model(5, True)
        # Solution gets .9351
        self.assertLessEqual(0.9349, accuracy)


def main() -> None:
    tests = [
        TestFeatureExtraction,
        TestNoAveraging,
        TestAveraging,
        # Should be last since it's slowest
        TestModelPredictions,
    ]
    grader = Grader(tests)
    grader.print_results()


if __name__ == "__main__":
    main()
