import pandas as pd
import sqlalchemy as sql
import numpy as np
import bokeh.plotting as bp
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, show


class DatabaseConnection:
    """Provides a connection to the database."""

    def __init__(self, db_name):
        """
        Initializes the DatabaseConnection object.

        Args:
            db_name: Name of the database.
        """
        self.engine = sql.create_engine(f'sqlite:///{db_name}.sqlite')

    def save_dataframe(self, df, table_name):
        """
        Saves a pandas DataFrame to the database as a table.

        Args:
            df: The DataFrame to be saved.
            table_name: Name of the table.
        """
        df.to_sql(table_name, self.engine, index=False, if_exists='replace')


class LeastSquaresError:
    """Calculates the least-squares error between two arrays of values."""

    @staticmethod
    def calculate(y_true, y_pred):
        """
        Calculates the least-squares error between two arrays of values.

        Args:
            y_true: An array of true values.
            y_pred: An array of predicted values.

        Returns:
            The least-squares error between the two arrays.
        """
        return np.sum((y_true - y_pred) ** 2)


class IdealFunctionMatcher:
    """Finds the ideal function that best matches the training data."""

    @staticmethod
    def find_best(train_data, ideal_df):
        """
        Finds the ideal function that best matches the training data.

        Args:
            train_data: A pandas DataFrame containing the training data.
            ideal_df: A pandas DataFrame containing the ideal functions.

        Returns:
            The index of the ideal function that best matches the training data.
        """
        least_squares_errors = []
        for i in range(1, len(ideal_df.columns)):
            ideal_function = ideal_df.iloc[:, i]
            least_squares_errors.append(LeastSquaresError.calculate(train_data['y'], ideal_function))

        best_index = np.argmin(least_squares_errors)
        return best_index + 1


class TestDataMatcher:
    """Matches the test data to the ideal functions."""

    @staticmethod
    def match(test_data, ideal_df):
        """
        Matches the test data to the ideal functions.

        Args:
            test_data: A pandas DataFrame containing the test data.
            ideal_df: A pandas DataFrame containing the ideal functions.

        Returns:
            A pandas DataFrame containing the matched test data.
        """
        best_ideal_functions = []
        for i in range(len(test_data)):
            best_ideal_function = IdealFunctionMatcher.find_best(test_data.iloc[i, 1:], ideal_df)
            best_ideal_functions.append(best_ideal_function)

        matched_test_data = pd.DataFrame({
            'x': test_data['x'],
            'y': test_data['y'],
            'ideal_function': best_ideal_functions
        })
        return matched_test_data


def create_scatter_plot(test_data, ideal_df):
    """
    Creates a scatter plot of the test data and the ideal functions.

    Args:
        test_data: A pandas DataFrame containing the test data.
        ideal_df: A pandas DataFrame containing the ideal functions.
    """
    plot = figure(title='Test Data', x_axis_label='x', y_axis_label='y')
    plot.scatter(test_data['x'], test_data['y'], legend_label='Test Data', color='red')

    for i in range(1, len(ideal_df.columns)):
        ideal_function = ideal_df.iloc[:, i]
        plot.line(ideal_df['x'], ideal_function, legend_label=f'Ideal Function {i}', line_width=2)

    show(plot)


class UnitTests:
    """Unit tests for the code elements."""

    @staticmethod
    def test_least_squares_error():
        """Unit test for the least_squares_error function."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])
        assert LeastSquaresError.calculate(y_true, y_pred) == 3

    @staticmethod
    def test_ideal_function_matcher():
        """Unit test for the IdealFunctionMatcher class."""
        train_data = pd.DataFrame({'y': [1, 2, 3, 4]})
        ideal_df = pd.DataFrame({'x': [0, 1, 2, 3], 'y1': [0, 1, 2, 3], 'y2': [1, 2, 3, 4]})
        assert IdealFunctionMatcher.find_best(train_data, ideal_df) == 2

    @staticmethod
    def test_test_data_matcher():
        """Unit test for the TestDataMatcher class."""
        test_data = pd.DataFrame({'x': [0, 1, 2, 3], 'y': [1, 2, 3, 4]})
        ideal_df = pd.DataFrame({'x': [0, 1, 2, 3], 'y1': [0, 1, 2, 3], 'y2': [1, 2, 3, 4]})
        matched_test_data = TestDataMatcher.match(test_data, ideal_df)
        assert len(matched_test_data) == 4
        assert set(matched_test_data.columns) == {'x', 'y', 'ideal_function'}

    @staticmethod
    def test_create_scatter_plot():
        """Unit test for the create_scatter_plot function."""
        test_data = pd.DataFrame({'x': [0, 1, 2, 3], 'y': [1, 2, 3, 4]})
        ideal_df = pd.DataFrame({'x': [0, 1, 2, 3], 'y1': [0, 1, 2, 3], 'y2': [1, 2, 3, 4]})
        create_scatter_plot(test_data, ideal_df)
        # Test if the plot is created without any errors (visual inspection needed)

    @staticmethod
    def run_all():
        """Runs all unit tests."""
        UnitTests.test_least_squares_error()
        UnitTests.test_ideal_function_matcher()
        UnitTests.test_test_data_matcher()
        UnitTests.test_create_scatter_plot()


# Run unit tests
UnitTests.run_all()

# Create a database connection
db_connection = DatabaseConnection('database')

# Save the dataframes to the database
db_connection.save_dataframe(pd.read_csv('train.csv'), 'train_data')
db_connection.save_dataframe(pd.read_csv('ideal.csv'), 'ideal_functions')
db_connection.save_dataframe(pd.read_csv('test.csv'), 'test_data')

# Read the dataframes from the database
train_df = pd.read_sql('train_data', db_connection.engine)
ideal_df = pd.read_sql('ideal_functions', db_connection.engine)
test_df = pd.read_sql('test_data', db_connection.engine)

# Match the test data to the ideal functions
matched_test_data = TestDataMatcher.match(test_df, ideal_df)

# Calculate the deviation between the test data and the ideal functions
deviation = matched_test_data['y'] - matched_test_data['ideal_function']

# Create and show the scatter plot
create_scatter_plot(matched_test_data, ideal_df)
