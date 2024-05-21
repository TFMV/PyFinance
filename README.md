# Value at Risk (VaR) Implementation

This project provides a Python implementation for calculating the Value at Risk (VaR) of a financial portfolio using historical stock data. The VaR calculation is based on the Variance-Covariance method.

![PyFinance](assets/PyFinance.webp)

## Features

- **Historical Data Fetching:** Uses `yfinance` to fetch historical stock data from Yahoo Finance.
- **Variance-Covariance VaR Calculation:** Calculates daily VaR at a specified confidence interval.

## Installation

To run this project, you'll need to have Python installed. It's recommended to use a virtual environment. Follow these steps to set up and run the project:

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/yourusername/PyFinance.git
    cd PyFinance
    ```

2. **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the Required Libraries:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Run the Script:**

You will want to update the portfolio value and perhaps confidence

Provide the ticker as an argument, like so

```bash
python VaR.py WFC
```

## Example Output

When you run the script, you should see an output similar to the following:

```bash
[*******************100%%********************] 1 of 1 completed
Ticker: WFC
Value-at-Risk: $36,958.04
```

## License

This project is licensed under the MIT License
