
import sys
import os
import pytest
import numpy as np
from math import pi, sin, cos
import matplotlib.pyplot as plt
from unittest.mock import call

"""
Note: code is for reference only (taken from an online course)
"""


if __name__ == '__main__':
    # Your first unit test using pytest:

    # Import the function convert_to_int()
    from preprocessing_helpers import convert_to_int

    # Complete the unit test name by adding a prefix
    def test_on_string_with_one_comma():
        # Complete the assert statement
        assert convert_to_int("2,081") == 2081

    ######################################################################
    # Spotting and fixing bugs:

    def convert_to_int(string_with_comma):
        # Fix this line so that it returns an int, not a str
        return int(string_with_comma.replace(",", ""))

    ######################################################################
    # Write an informative test failure message:

    from preprocessing_helpers import convert_to_int

    def test_on_string_with_one_comma():
        test_argument = "2,081"
        expected = 2081
        actual = convert_to_int(test_argument)
        # Format the string with the actual return value
        message = ("convert_to_int('2,081') should return the int 2081, but "
                   "it actually returned {0}".format(actual))
        # Write the assert statement which prints message on failure
        assert actual == expected, message

    ######################################################################
    # Testing float return values:

    from as_numpy import get_data_as_numpy_array

    def test_on_clean_file():
        expected = np.array([[2081.0, 314942.0],
                             [1059.0, 186606.0],
                             [1148.0, 206186.0]
                             ]
                            )
        actual = get_data_as_numpy_array("example_clean_data.txt",
                                         num_columns=2)
        message = "Expected return value: {0}, Actual return value: {1}".format(
            expected, actual)
        # Complete the assert statement
        assert actual == pytest.approx(expected), message

    ######################################################################
    # Testing with multiple assert statements:

    def test_on_six_rows():
        example_argument = np.array([[2081.0, 314942.0], [1059.0, 186606.0],
                                     [1148.0, 206186.0], [1506.0, 248419.0],
                                     [1210.0, 214114.0], [1697.0, 277794.0]]
                                    )
        # Fill in with training array's expected number of rows
        expected_training_array_num_rows = 4
        # Fill in with testing array's expected number of rows
        expected_testing_array_num_rows = 2
        actual = split_into_training_and_testing_sets(example_argument)
        # Write the assert statement checking training array's number of rows
        assert (actual[0].shape[0] == expected_training_array_num_rows,
                "The actual number of rows in the training array is not {}"
                .format(expected_training_array_num_rows))
        # Write the assert statement checking testing array's number of rows
        assert (actual[1].shape[0] == expected_testing_array_num_rows,
                "The actual number of rows in the testing array is not "
                "{}".format(expected_testing_array_num_rows))

    ######################################################################
    # Practice the context manager:

    # Fill in with a context manager that will silence the ValueError
    with pytest.raises(ValueError):
        raise ValueError

    ######################################################################

    try:
        # Fill in with a context manager that raises Failed if no OSError is
        # raised
        with pytest.raises(OSError):
            raise ValueError
    except:
        print("pytest raised an exception because no OSError was raised in "
              "the context.")

    ######################################################################

    # Store the raised ValueError in the variable exc_info
    with pytest.raises(ValueError) as exc_info:
        raise ValueError("Silence me!")

    ######################################################################

    with pytest.raises(ValueError) as exc_info:
        raise ValueError("Silence me!")
    # Check if the raised ValueError contains the correct message
    assert exc_info.match("Silence me!")

    ######################################################################
    # Unit test a ValueError:

    from train import split_into_training_and_testing_sets

    def test_on_one_row():
        test_argument = np.array([[1382.0, 390167.0]])
        # Store information about raised ValueError in exc_info
        with pytest.raises(ValueError) as exc_info:
            split_into_training_and_testing_sets(test_argument)
        expected_error_msg = ("Argument data_array must have at least 2 rows, "
                              "it actually has just 1")
        # Check if the raised ValueError contains the correct message
        assert exc_info.match(expected_error_msg)

    ######################################################################
    # Testing well: Boundary values:

    from preprocessing_helpers import row_to_list

    def test_on_no_tab_no_missing_value():  # (0, 0) boundary value
        # Assign actual to the return value for the argument "123\n"
        actual = row_to_list("123\n")
        assert actual is None, "Expected: None, Actual: {0}".format(actual)

    def test_on_two_tabs_no_missing_value():  # (2, 0) boundary value
        actual = row_to_list("123\t4,567\t89\n")
        # Complete the assert statement
        assert actual is None, "Expected: None, Actual: {0}".format(actual)

    def test_on_one_tab_with_missing_value():  # (1, 1) boundary value
        actual = row_to_list("\t4,567\n")
        # Format the failure message
        assert actual is None, "Expected: None, Actual: {0}".format(actual)

    ######################################################################
    # Testing well: Values triggering special logic:

    from preprocessing_helpers import row_to_list

    def test_on_no_tab_with_missing_value():  # (0, 1) case
        # Assign to the actual return value for the argument "\n"
        actual = row_to_list("\n")
        # Write the assert statement with a failure message
        assert actual is None, "Expected: None, Actual: {0}".format(actual)

    def test_on_two_tabs_with_missing_value():  # (0, 1) case
        # Assign to the actual return value for the argument "123\t\t89\n"
        actual = row_to_list("123\t\t89\n")
        # Write the assert statement with a failure message
        assert actual is None, "Expected: None, Actual: {0}".format(actual)

    ######################################################################
    # Testing well: Normal arguments:

    from preprocessing_helpers import row_to_list

    def test_on_normal_argument_1():
        actual = row_to_list("123\t4,567\n")
        # Fill in with the expected return value for the argument
        # "123\t4,567\n"
        expected = ["123", "4,567"]
        assert actual == expected, "Expected: {0}, Actual: {1}".format(
            expected, actual)

    def test_on_normal_argument_2():
        actual = row_to_list("1,059\t186,606\n")
        expected = ["1,059", "186,606"]
        # Write the assert statement along with a failure message
        assert actual == expected, "Expected: {0}, Actual: {1}".format(
            expected, actual)

    ######################################################################
    # TDD: Tests for normal arguments:

    # In this and the following exercises, you will implement the function
    # convert_to_int() using Test Driven Development (TDD). In TDD, you write
    # the tests first and implement the function later.
    #
    # Normal arguments for convert_to_int() are integer strings with comma as
    # thousand separators. Since the best practice is to test a function for
    # two to three normal arguments, here are three examples with no comma,
    # one comma and two commas respectively.

    # Since the convert_to_int() function does not exist yet, you won't be
    # able to import it. But you will use it in the tests anyway. That's how
    # TDD works.

    def test_with_no_comma():
        actual = convert_to_int("756")
        # Complete the assert statement
        assert actual == 756, "Expected: 756, Actual: {0}".format(actual)

    def test_with_one_comma():
        actual = convert_to_int("2,081")
        # Complete the assert statement
        assert actual == 2081, "Expected: 2081, Actual: {0}".format(actual)

    def test_with_two_commas():
        actual = convert_to_int("1,034,891")
        # Complete the assert statement
        assert actual == 1034891, "Expected: 1034891, Actual: {0}".format(
            actual)

    ######################################################################
    # TDD: Requirement collection:

    # Give a name to the test for an argument with missing comma
    def test_on_string_with_missing_comma():
        actual = convert_to_int("178100,301")
        assert actual is None, "Expected: None, Actual: {0}".format(actual)

    def test_on_string_with_incorrectly_placed_comma():
        # Assign to the actual return value for the argument "12,72,891"
        actual = convert_to_int("12,72,891")
        assert actual is None, "Expected: None, Actual: {0}".format(actual)

    def test_on_float_valued_string():
        actual = convert_to_int("23,816.92")
        # Complete the assert statement
        assert actual is None, "Expected: None, Actual: {0}".format(actual)

    ######################################################################
    # TDD: Implement the function:

    def convert_to_int(integer_string_with_commas):
        comma_separated_parts = integer_string_with_commas.split(",")
        for i in range(len(comma_separated_parts)):
            # Write an if statement for checking missing commas
            if len(comma_separated_parts[i]) > 3:
                return None
            # Write the if statement for incorrectly placed commas
            if i != 0 and len(comma_separated_parts[i]) != 3:
                return None
        integer_string_without_commas = "".join(comma_separated_parts)
        try:
            return int(integer_string_without_commas)
        # Fill in with the correct exception for float valued argument strings
        except ValueError:
            return None

    ######################################################################
    # Create a test class:

    # Test classes are containers inside test modules, and serve as a
    # structuring tool in the pytest framework. Within the test module, they
    # help separate tests for different functions.

    from models.train import split_into_training_and_testing_sets

    # Declare the test class
    class TestSplitIntoTrainingAndTestingSets(object):
        # Fill in with the correct mandatory argument
        def test_on_one_row(self):
            test_argument = np.array([[1382.0, 390167.0]])
            with pytest.raises(ValueError) as exc_info:
                split_into_training_and_testing_sets(test_argument)
            expected_error_msg = ("Argument data_array must have at least 2 "
                                  "rows, it actually has just 1")
            assert exc_info.match(expected_error_msg)

    ######################################################################
    # One command to run them all:

    # Assuming that you simply want to answer the binary question "Are all
    # tests passing" without wasting time and resources, what is the correct
    # command to run all tests till the first failure is encountered?

    # In command line:
    #   pytest -x

    ######################################################################
    # Running test classes:

    def split_into_training_and_testing_sets(data_array):
        dim = data_array.ndim
        if dim != 2:
            raise ValueError(
                "Argument data_array must be two dimensional. Got {0} "
                "dimensional array instead!".format(dim))
        num_rows = data_array.shape[0]
        if num_rows < 2:
            raise ValueError(
                "Argument data_array must have at least 2 rows, it actually "
                "has just {0}".format(num_rows))
        # Fill in with the correct float
        num_training = int(0.75 * data_array.shape[0])
        permuted_indices = np.random.permutation(data_array.shape[0])
        return (data_array[permuted_indices[:num_training], :],
                data_array[permuted_indices[num_training:], :])

    # What is the correct command to run all the tests in this test class
    # using node IDs?

    # In command line:
    #   pytest models/test_train.py::TestSplitIntoTrainingAndTestingSets

    # What is the correct command to run only the previously failing test test_
    # on_six_rows() using node IDs?

    # In command line:
    #   pytest models/test_train.py::TestSplitIntoTrainingAndTestingSets::test_on_six_rows

    # What is the correct command to run the tests in
    # TestSplitIntoTrainingAndTestingSets using keyword expressions?

    # In command line:
    #   pytest -k "SplitInto"

    ######################################################################
    # Mark a test class as expected to fail:

    # A new function model_test() is being developed and it returns the
    # accuracy of a given linear regression model on a testing dataset. Test
    # Driven Development (TDD) is being used to implement it. The procedure
    # is: write tests first and then implement the function.

    # A test class TestModelTest has been created within the test module
    # models/test_train.py. In the test class, there are two unit tests called
    # test_on_linear_data() and test_on_one_dimensional_array(). But the
    # function model_test() has not been implemented yet.

    # Add a reason for the expected failure
    @pytest.mark.xfail(
        reason="Using TDD, model_test() has not yet been implemented")
    class TestModelTest(object):
        def test_on_linear_data(self):
            test_input = np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
            expected = 1.0
            actual = model_test(test_input, 2.0, 1.0)
            message = ("model_test({0}) should return {1}, but it actually "
                       "returned {2}".format(test_input, expected, actual))
            assert actual == pytest.approx(expected), message

        def test_on_one_dimensional_array(self):
            test_input = np.array([1.0, 2.0, 3.0, 4.0])
            with pytest.raises(ValueError) as exc_info:
                model_test(test_input, 1.0, 1.0)

    ######################################################################
    # Mark a test as conditionally skipped:

    # In Python 2, there was a built-in function called xrange(). In Python 3,
    # xrange() was removed. Therefore, if any test uses xrange(), it's going
    # to fail with a NameError in Python 3.
    #
    # Remember the function get_data_as_numpy_array()? You saw it in Chapter
    # 2. It converted data in a preprocessed data file into a NumPy array.
    #
    # range() has been deliberately replaced with the obsolete xrange() in the
    # function. Evil laughter! But no worries, it will be changed back after
    # you're done with this exercise.

    class TestGetDataAsNumpyArray(object):
        # Add a reason for skipping the test
        @pytest.mark.skipif(sys.version_info > (2, 7),
                            reason="Works only on Python 2.7 or lower")
        def test_on_clean_file(self):
            expected = np.array([[2081.0, 314942.0],
                                 [1059.0, 186606.0],
                                 [1148.0, 206186.0]
                                 ]
                                )
            actual = get_data_as_numpy_array("example_clean_data.txt",
                                             num_columns=2)
            message = ("Expected return value: {0}, Actual return value: {1}"
                .format(expected, actual))
            assert actual == pytest.approx(expected), message

    ######################################################################
    # Reasoning in the test result report:

    # What is the command that would only show the reason for expected
    # failures in the test result report?

    # In command line:
    #   pytest -rx

    # What is the command that would only show the reason for skipped tests in
    # the test result report?

    # In command line:
    #   pytest -rs

    # What is the command that would show the reason for both skipped tests
    # and tests that are expected to fail in the test result report?

    # In command line:
    #   pytest -rsx

    ######################################################################
    # Continuous integration and code coverage:

    # CI server - "Travis CI" (they have a restricted free service for open
    # source projects) - to set up the automatic running of unit tests on each
    # commit to GitHub.
    # https://docs.travis-ci.com/user/for-beginners/

    # For Travis CI, require a file at the highest level of the repo called
    # ".travis.yml":

    ### .travis.yml ###
    # language: python
    # python:
    #   - "3.6"
    # install:
    #   - pip install -e .
    # script:
    #   - pytest tests
    ######

    # In command line (push to GitHub):
    #   git add .travis.yml
    #   git push origin master

    # 'Build Passing' badge:
    # On the GitHub website, go to 'Marketplace', install 'Travis CI'
    # (selecting the repos to use it with), and sign into Travis with your
    # GitHub. Once Travis finishes running the tests on their servers, the
    # "build passing" badge appears next to the name of the repo of GitHub.
    # Click on the badge and select 'MARKDOWN', copy and paste the result into
    # the README.md file to display it.

    ### .travis.yml ###
    # language: python
    # python:
    #   - "3.6"
    # install:
    #   - pip install -e .
    #   - pip install pytest-cov codecov    # Install packages for code coverage report
    # script:
    #   - pytest --cov=src tests            # Point to the source directory
    # after_success:
    #   - codecov                           # uploads report to codecov.io
    ######

    # 'Code Coverage' badge:
    # Install 'Codecov' from the 'Marketplace'. Commits then lead to coverage
    # report at 'codecov.io'. Get the Markdown for the badge (from the
    # 'Settings' of Codecov), copy and paste the markdown into the README.md
    # of the repo on GitHub.

    ######################################################################
    # Use a fixture for a clean data file:

    # In the video, you saw how the preprocess() function creates a clean data
    # file.
    #
    # The get_data_as_numpy_array() function takes the path to this clean data
    # file as the first argument and the number of columns of data as the
    # second argument. It returns a NumPy array holding the data.
    #
    # In a previous exercise, you wrote the test test_on_clean_file() without
    # using a fixture. That's bad practice! This time, you'll use the fixture
    # clean_data_file(), which
    # - creates a clean data file in the setup,
    # - yields the path to the clean data file,
    # - removes the clean data file in the teardown.

    # Add a decorator to make this function a fixture
    @pytest.fixture
    def clean_data_file():
        file_path = "clean_data_file.txt"
        with open(file_path, "w") as f:
            f.write("201\t305671\n7892\t298140\n501\t738293\n")
        yield file_path
        os.remove(file_path)

    # Pass the correct argument so that the test can use the fixture
    def test_on_clean_file(clean_data_file):
        expected = np.array(
            [[201.0, 305671.0], [7892.0, 298140.0], [501.0, 738293.0]])
        # Pass the clean data file path yielded by the fixture as the first
        # argument
        actual = get_data_as_numpy_array(clean_data_file, 2)
        assert actual == pytest.approx(
            expected), "Expected: {0}, Actual: {1}".format(expected, actual)

    ######################################################################
    # Write a fixture for an empty data file:

    # When a function takes a data file as an argument, you need to write a
    # fixture that takes care of creating and deleting that data file. This
    # exercise will test your ability to write such a fixture.
    #
    # get_data_as_numpy_array() should return an empty numpy array if it gets
    # an empty data file as an argument. To test this behavior, you need to
    # write a fixture empty_file() that does the following.
    #
    # - Creates an empty data file empty.txt relative to the current working
    # directory in setup.
    # - Yields the path to the empty data file.
    # - Deletes the empty data file in teardown.

    @pytest.fixture
    def empty_file():
        # Assign the file path "empty.txt" to the variable
        file_path = "empty.txt"
        open(file_path, "w").close()
        # Yield the variable file_path
        yield file_path
        # Remove the file in the teardown
        os.remove(file_path)

    def test_on_empty_file(self, empty_file):
        expected = np.empty((0, 2))
        actual = get_data_as_numpy_array(empty_file, 2)
        assert actual == pytest.approx(
            expected), "Expected: {0}, Actual: {1}".format(expected, actual)

    ######################################################################
    # Fixture chaining using tmpdir:

    # The built-in tmpdir fixture is very useful when dealing with files in
    # setup and teardown. tmpdir combines seamlessly with user defined fixture
    # via fixture chaining.
    #
    # In this exercise, you will use the power of tmpdir to redefine and
    # improve the empty_file() fixture that you wrote in the last exercise and
    # get some experience with fixture chaining.

    @pytest.fixture
    # Add the correct argument so that this fixture can chain with the tmpdir
    # fixture
    def empty_file(tmpdir):
        # Use the appropriate method to create an empty file in the temporary
        # directory
        file_path = tmpdir.join("empty.txt")
        open(file_path, "w").close()
        yield file_path

    # In what order will the setup and teardown of empty_file() and tmpdir be
    # executed?

    # setup of tmpdir → setup of empty_file() → teardown of empty_file() →
    # teardown of tmpdir.

    ######################################################################
    # Program a bug-free dependency:

    # In the video, row_to_list() was mocked. But preprocess() has another
    # dependency convert_to_int(). Generally, its best to mock all
    # dependencies of the function under test. It's your job to mock convert_
    # to_int() in this and the following exercises.

    # Define a function convert_to_int_bug_free
    def convert_to_int_bug_free(comma_separated_integer_string):
        # Assign to the dict holding the correct return values
        return_values = {
            "1,801": 1801, "201,411": 201411, "2,002": 2002,
            "333,209": 333209, "1990": None, "782,911": 782911,
            "1,285": 1285, "389129": None}
        # Return the correct result using the dict return_values
        return return_values[comma_separated_integer_string]

    ######################################################################
    # Mock a dependency:

    # Mocking helps us replace a dependency with a MagicMock() object.
    # Usually, the MagicMock() is programmed to be a bug-free version of the
    # dependency. To verify whether the function under test works properly
    # with the dependency, you simply check whether the MagicMock() is called
    # with the correct arguments and in the right order.
    #
    # In the last exercise, you programmed a bug-free version of the
    # dependency data.preprocessing_helpers.convert_to_int in the context of
    # the test test_on_raw_data(), which applies preprocess() on a raw data
    # file. The data file is printed out in the IPython console.

    # Add the correct argument to use the mocking fixture in this test
    def test_on_raw_data(self, raw_and_clean_data_file, mocker):
        raw_path, clean_path = raw_and_clean_data_file
        # Replace the dependency with the bug-free mock
        convert_to_int_mock = mocker.patch(
            "data.preprocessing_helpers.convert_to_int",
            side_effect=convert_to_int_bug_free)
        preprocess(raw_path, clean_path)
        # Check if preprocess() called the dependency correctly
        assert convert_to_int_mock.call_args_list == [
            call("1,801"), call("201,411"), call("2,002"), call("333,209"),
            call("1990"), call("782,911"), call("1,285"), call("389129")]
        with open(clean_path, "r") as f:
            lines = f.readlines()
        first_line = lines[0]
        assert first_line == "1801\\t201411\\n"
        second_line = lines[1]
        assert second_line == "2002\\t333209\\n"

    ######################################################################
    # Testing on linear data:

    # The model_test() function, which measures how well the model fits unseen
    # data, returns a quantity called r2 which is very difficult to compute in
    # the general case. Therefore, you need to find special testing sets where
    # computing r2 is easy.
    #
    # One important special case is when the model fits the testing set
    # perfectly. This means that all the data points fall exactly on the best
    # fit line. In other words, the testing set is perfectly linear. One such
    # testing set is printed out in the IPython console for you.
    #
    # In this special case, model_test() should return 1.0 if the model's
    # slope and intercept matches that of the testing set, because 1.0 is
    # usually highest possible value that r2 can take.

    from models.train import model_test

    def test_on_perfect_fit():
        # Assign to a NumPy array containing a linear testing set
        test_argument = np.array([[1., 3.], [2., 5.], [3., 7.]])
        # Fill in with the expected value of r^2 in the case of perfect fit
        expected = 1.
        # Fill in with the slope and intercept of the model
        actual = model_test(test_argument, slope=2., intercept=1.)
        # Complete the assert statement
        assert actual == pytest.approx(
            expected), "Expected: {0}, Actual: {1}".format(expected, actual)

    ######################################################################
    # Testing on circular data:

    # Another special case where it is easy to guess the value of r2 is when
    # the model does not fit the testing dataset at all. In this case, r2
    # takes its lowest possible value 0.0.
    #
    # The plot shows such a testing dataset and model. The testing dataset
    # consists of data arranged in a circle of radius 1.0. The x and y
    # co-ordinates of the data is shown on the plot. The model corresponds to
    # a straight line y=0.
    #
    # As one can easily see, the straight line does not fit the data at all.
    # In this particular case, the value of r2 is known to be 0.0.

    def test_on_circular_data(self):
        theta = pi / 4.0
        # Assign to a NumPy array holding the circular testing data
        test_argument = np.array([[1.0, 0.0], [cos(theta), sin(theta)],
                                  [0.0, 1.0],
                                  [cos(3 * theta), sin(3 * theta)],
                                  [-1.0, 0.0],
                                  [cos(5 * theta), sin(5 * theta)],
                                  [0.0, -1.0],
                                  [cos(7 * theta), sin(7 * theta)]]
                                 )
        # Fill in with the slope and intercept of the straight line
        actual = model_test(test_argument, slope=0.0, intercept=0.0)
        # Complete the assert statement
        assert actual == pytest.approx(0.)

    ######################################################################
    # Generate the baseline image:

    # In this exercise, you will test the function introduced in the video
    # get_plot_for_best_fit_line() on another set of test arguments. Here is
    # the test data.
    #
    # 1.0    3.0
    # 2.0    8.0
    # 3.0    11.0
    #
    # The best fit line that the test will draw follows the equation y=5x−2.
    # Two points, (1.0, 3.0) and (2.0, 8.0) will fall on the line. The point
    # (3.0, 11.0) won't. The title of the plot will be "Test plot for almost
    # linear data".

    from visualization.plots import get_plot_for_best_fit_line

    class TestGetPlotForBestFitLine(object):
        # Add the pytest marker which generates baselines and compares images
        @pytest.mark.mpl_image_compare
        def test_plot_for_almost_linear_data(self):
            slope = 5.0
            intercept = -2.0
            x_array = np.array([1.0, 2.0, 3.0])
            y_array = np.array([3.0, 8.0, 11.0])
            title = "Test plot for almost linear data"
            # Return the matplotlib figure returned by the function under test
            return get_plot_for_best_fit_line(slope, intercept, x_array,
                                              y_array, title)

    # In command line:
    #   pip install pytest-mpl
    #   pytest -k "TestGetPlotForBestFitLine" --mpl-generate-path project/tests/visualization/baseline

    ######################################################################
    # Run the tests for the plotting function:

    # Shortly after the baseline image was generated, one of your colleagues
    # modified the plotting function. You have to run the tests in order to
    # check whether the function still works as expected.
    #
    # Remember that the tests were housed in a test class
    # TestGetPlotForBestFitLine in the test module visualization/test_plots.py.

    # In command line:
    #   pytest -k "TestGetPlotForBestFitLine" --mpl

    ######################################################################
    # Fix the plotting function:

    # In the last exercise, pytest saved the baseline images, actual images,
    # and images containing the pixelwise difference in a temporary folder.
    # The difference image for one of the tests test_on_almost_linear_data()
    # is shown below.

    def get_plot_for_best_fit_line(slope, intercept, x_array, y_array, title):
        fig, ax = plt.subplots()
        ax.plot(x_array, y_array, ".")
        ax.plot([0, np.max(x_array)],
                [intercept, slope * np.max(x_array) + intercept], "-")
        # Fill in with axis labels so that they match the baseline
        ax.set(xlabel='area (square feet)', ylabel='price (dollars)',
               title=title)
        return fig

    ######################################################################
