"""Grading system based on unittest test cases."""

# Copyright 2021 Constantine Lignos and Ceasar Bautista
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import inspect
import sys
import threading
import time
import unittest
from functools import partial
from typing import Any, Callable, Union, TextIO, Optional, Iterable, Type

_POINTS_ATTRIBUTE = "_points"
_TIMEOUT_ATTRIBUTE = "_timeout"


def points(n: Union[float, int]) -> partial:
    """Decorator used to add a _points attribute to an object."""
    return partial(_add_attr, attr=_POINTS_ATTRIBUTE, value=n)


def timeout(n: Union[float, int]) -> partial:
    """Decorator used to add a _timeout attribute to an object."""
    return partial(_add_attr, attr=_TIMEOUT_ATTRIBUTE, value=n)


def _add_attr(obj: Any, attr: str, value: Any) -> Any:
    old_points = getattr(obj, attr, None)
    assert not old_points, f"Object already has a {attr} attribute"
    setattr(obj, attr, value)
    return obj


class Problem:
    """A Problem that can be graded.

    test_case should be an instance of unittest.TestCase

    test_weights should be a list of test_name-weight pairs.

    timeout should be the time to wait before killing a test, specified in
    seconds. By default, timeout is None and the test will wait until
    completion."""

    def __init__(
        self, test_case: Type[unittest.TestCase], timeout: Union[float, int] = None
    ) -> None:
        self.timeout: Optional[Union[float, int]] = timeout
        self._test_case: Type[unittest.TestCase] = test_case
        self._results: dict[Callable, unittest.TestResult] = {}

        # Check each test method to get points
        self.test_weights: dict[Callable, float] = {}
        for name, attr in test_case.__dict__.items():
            if name.startswith("test") and inspect.isfunction(attr):
                points = getattr(attr, "_points", None)
                if points is None:
                    raise ValueError(f"Test method {name} has no points set")
                else:
                    self.test_weights[attr] = float(points)

        # We have to manually run class setup here as it's up to the runner, unlike
        # the instance setUp method
        test_case.setUpClass()

    def run_tests(self, log_file: TextIO) -> float:
        """Run tests, populate results, and return the grade."""
        print(f"Grading {self._test_case.__name__}", file=log_file)  # type: ignore
        print(file=log_file)
        for test, weight in self.test_weights.items():
            print(f"Running {test.__name__}", file=log_file)

            if test.__doc__:
                print(test.__doc__, file=log_file)
            start_time = time.perf_counter()
            result = self.run(test)
            elapsed = time.perf_counter() - start_time
            if result.wasSuccessful():
                if hasattr(test, _TIMEOUT_ATTRIBUTE):
                    limit = getattr(test, _TIMEOUT_ATTRIBUTE)
                    print(
                        f"Time: {elapsed:0.3f} seconds (limit {limit:0.3f})",
                        file=log_file,
                    )
                print(f"Points: {weight}/{weight}", file=log_file)
            else:
                print(file=log_file)
                print(
                    "Test failed with the error below, displayed between lines of ---.",
                    file=log_file,
                )
                print(
                    "The expected value is given first, followed by the actual result.",
                    file=log_file,
                )
                print("-" * 70, file=log_file)
                # Get the error/failure
                try:
                    print(result.errors[0][1], file=log_file)
                except IndexError:
                    pass
                try:
                    print(result.failures[0][1], file=log_file)
                except IndexError:
                    pass
                print("-" * 70, file=log_file)
                print(f"Points: {0.0}/{weight}", file=log_file)
            print(file=log_file)
        print("=" * 70, file=log_file)
        return self.grade

    def run(self, test_method: Callable) -> unittest.TestResult:
        """Return the result for the given test."""
        # TestCase supports getting a runner for an individual method by name
        test = self._test_case(test_method.__name__)  # type: ignore
        result = unittest.TestResult()
        test_runner = threading.Thread(target=test.run, args=(result,))  # type: ignore
        test_runner.daemon = True
        test_runner.start()

        # Override global timeout based on annotation
        test_timeout = getattr(test_method, _TIMEOUT_ATTRIBUTE, self.timeout)
        test_runner.join(test_timeout)

        # If the test is still running, report a failure
        if test_runner.is_alive():
            # Create a fake exception so we can use that for the failure
            try:
                raise TimeoutError(
                    f"Test {repr(test)} took longer than {test_timeout} seconds"
                )
            except TimeoutError:
                info = sys.exc_info()
            result.addFailure(test, info)  # type: ignore

        self._results[test_method] = result
        return result

    @property
    def grade(self) -> float:
        """Grade earned for the problem."""
        assert self._results, "Tests have not been run"
        return sum(
            weight
            for test, weight in self.test_weights.items()
            if self._results[test].wasSuccessful()
        )

    @property
    def max_grade(self) -> float:
        """The maximum grade possible for the problem."""
        return sum(self.test_weights.values())


class Grader(object):
    """A grader object."""

    def __init__(
        self, test_classes: Iterable[Type[unittest.TestCase]], **kwargs: Any
    ) -> None:
        self.problems = [Problem(test_class, **kwargs) for test_class in test_classes]

    def print_results(self, log_file: TextIO = sys.stdout) -> None:
        """Grade each problem and print out the final grade."""
        print("=" * 70, file=log_file)
        total = 0.0
        max_points = 0.0
        start_time = time.perf_counter()
        for problem in self.problems:
            total += problem.run_tests(log_file)
            max_points += problem.max_grade
        elapsed = time.perf_counter() - start_time
        print(f"Total Time: {elapsed:0.1f} seconds", file=log_file)
        print(f"Total Grade: {total}/{max_points}", file=log_file)
