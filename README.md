# Bedrock Data Extractor

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An interactive Python application for extracting structured data (JSON, CSV) from unstructured text using Amazon Bedrock and LangChain. This tool provides a simple UI within a Jupyter environment to run queries against the Cohere Command model.

## Features

-   **Unstructured to Structured:** Convert plain text into well-formatted JSON or CSV.
-   **Powered by Amazon Bedrock:** Leverages the power of foundation models for reliable data extraction.
-   **Interactive UI:** Uses `ipywidgets` to provide a user-friendly interface for entering prompts and viewing results directly in a Jupyter Notebook or compatible IDE.
-   **Format Selection:** Easily switch between JSON and CSV as the desired output format.
-   **Error Handling:** Displays raw model output if parsing fails, helping with debugging prompts.

## Prerequisites

Before you begin, ensure you have the following:

1.  **Python 3.9+**
2.  **An AWS Account** with access to Amazon Bedrock. You must [request access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html) to the models you intend to use (e.g., Cohere Command).
3.  **AWS Credentials Configured:** Your AWS credentials must be configured locally. The recommended way is to install the AWS CLI and run `aws configure`.

    ```bash
    pip install awscli
    aws configure
    ```

    This will store your credentials in `~/.aws/credentials`, which the script will use automatically.

## Setup & Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Pavan164-ml/bedrock-data-extractor.git
    cd bedrock-data-extractor
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**
    A `requirements.txt` file is provided for easy installation.
    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should contain:
    ```
    boto3
    pandas
    langchain-community
    ipywidgets
    ipython
    jupyter
    ```

## Usage

This application is designed to be run in an environment that supports `ipywidgets`, such as:
-   Jupyter Notebook or JupyterLab
-   VS Code with the Jupyter extension
-   Google Colab

1.  **Start Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
2.  **Run the application script:**
    You can either paste the code from `app.py` into a notebook cell and run it, or you can run the file directly from a cell using the `%run` magic command:
    ```python
    %run app.py
    ```
3.  **Interact with the UI:**
    -   **Prompt:** Enter the text you want to process and the instructions for the model.
    -   **Output Format:** Select either JSON or CSV.
    -   **Extract Data:** Click the button to send the request to Bedrock.
    -   The results will be displayed in the output area below the button.

## Example Prompts

### JSON Extraction

**Prompt:**
```
From the text below, extract the user's full name, email address, and the total order amount as a number.

User profile: "Name: Jane Doe, Contact: jane.d@example.com. She recently placed an order for $149.99."
```
**Expected Output (JSON):**
```json
{
  "full_name": "Jane Doe",
  "email": "jane.d@example.com",
  "order_amount": 149.99
}
```

### CSV Extraction

**Prompt:**
```
Please list the following products in a table with columns for Product Name, Price, and Stock.

- The new A-series laptop is priced at $1200 and we have 50 units in stock.
- The B-series monitor costs $400 and there are 120 units available.
- The C-series keyboard is $75 with 300 units in stock.
```
**Expected Output (CSV):**
A Pandas DataFrame table will be displayed with the following data:
| Product Name | Price | Stock |
| :--- | :--- | :--- |
| A-series laptop | 1200 | 50 |
| B-series monitor| 400 | 120 |
| C-series keyboard| 75 | 300 |

## How It Works

1.  **UI Interaction:** The `ipywidgets` UI captures the user's prompt and desired output format.
2.  **Prompt Engineering:** The script appends a clear instruction to the user's prompt, guiding the model to produce output in the selected format (`JSON` or `CSV`).
3.  **Bedrock Invocation:** The `langchain_community` library sends the formatted prompt to the specified Amazon Bedrock model (`cohere.command-text-v14`).
4.  **Response Parsing:** The script receives the raw text response from the model and attempts to parse it.
    -   For **JSON**, it uses `json.loads()`.
    -   For **CSV**, it uses `pandas.read_csv()` via an in-memory `StringIO` buffer.
5.  **Display Results:** If parsing is successful, the structured data is displayed in a clean format (pretty-printed JSON or an HTML table for the DataFrame). If parsing fails, an error message and the raw model output are shown for debugging.
