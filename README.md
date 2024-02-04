# DataModules
A repository of Python modules designed specifically for data analysis and manipulation

# PandasEnhancer

`PandasEnhancer` is a Python module designed to extend the capabilities of the pandas library, making data manipulation and analysis more efficient and intuitive. By leveraging `pandas_flavor`, it registers additional methods directly on pandas DataFrame objects, facilitating a more seamless and integrated workflow for data scientists and analysts.

## Features

`PandasEnhancer` includes a variety of methods to enhance data manipulation, including but not limited to:

- Data cleaning and preprocessing (e.g., removing duplicates, filling missing values).
- Feature engineering (e.g., normalization, polynomial feature creation).
- Data exploration tools (e.g., generating summary statistics, correlation heatmaps).
- Visualization capabilities directly from DataFrame objects (e.g., histograms, scatter plots, correlation heatmaps).
- Model preparation utilities (e.g., encoding categorical variables, data splitting).

## Installation

Ensure you have `pandas` and `pandas_flavor` installed in your Python environment. You can install these dependencies using pip:

```sh
pip install pandas pandas_flavor
```

Then, install PandasEnhancer:

```sh
pip install git+https://github.com/yourusername/PandasEnhancer.git
```

Replace https://github.com/yourusername/PandasEnhancer.git with the actual URL of your GitHub repository.

## Usage

After installation, you can start using PandasEnhancer by importing the DataFrameManipulator class and applying its methods to your pandas DataFrames:

```python
import pandas as pd
from PandasEnhancer import DataFrameManipulator

# Sample data
data = {'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']}
df = pd.DataFrame(data)

# Initialize the manipulator
df_manipulator = DataFrameManipulator(df)

# Use the methods
df_clean = df_manipulator.remove_duplicates()
df_norm = df_manipulator.normalize('Column1')
```

## Contributing

Contributions to PandasEnhancer are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request. Please ensure your code adheres to the project's coding standards and includes tests covering new functionality or fixes.

## License

PandasEnhancer is released under the MIT License. See the LICENSE file in the repository for full license text.

### Notes for Customization:

- **Installation Command**: Update the installation command once your package is available for installation, whether through PyPI or directly from a GitHub URL as shown.
- **Usage Examples**: Provide more detailed examples tailored to the specific functionalities of your module to help users get started quickly.
- **Contributing Guidelines**: If your project is open to contributions, consider adding a `CONTRIBUTING.md` file to your repository with detailed guidelines on how to contribute.
- **License**: Ensure the `LICENSE` file exists in your repository with the appropriate licensing information. The MIT License is commonly used for open-source software, but you should choose the license that best fits your project's needs.

## Documentation

For detailed documentation on all methods and their parameters, please refer to the inline documentation within the code. Each method in the `DataFrameManipulator` class is accompanied by docstrings that explain the functionality, parameters, and return values.

## Examples

Below are some additional examples illustrating the use of `PandasEnhancer` for common data manipulation tasks:

### Filling Missing Values

```python
# Fill missing values in the DataFrame with zeros
df_filled = df_manipulator.fill_na(0)
```

### Encoding Categorical variables

```python
# Encode a categorical column into numeric labels
df_encoded = df_manipulator.encode('Column2')
```

### Plotting a Histogram

```python
# Plot a histogram of a numeric column
df_manipulator.plot_histogram('Column1')
```

### Removing Outliers

```python
# Remove outliers from a specific column
df_no_outliers = df_manipulator.remove_outliers('Column1')
```

## Advanced Usage

PandasEnhancer can also be extended with custom methods by contributing to the project. If you have a common data manipulation pattern that you believe could benefit others, consider implementing it and submitting a pull request.

## Support

If you encounter any issues or have questions about using PandasEnhancer, please open an issue on the GitHub repository. The community and maintainers are here to help.

## Acknowledgements

PandasEnhancer was created to simplify and streamline the process of data manipulation and analysis in Python. Thanks to all contributors and users for their support and feedback.

## See Also

- [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [pandas_flavor documentation](https://github.com/Zsailer/pandas_flavor)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.