# app.py

"""
An interactive Jupyter/IPython application to extract structured data (JSON or CSV)
from unstructured text using Amazon Bedrock and LangChain.
"""

import os
import json
import pandas as pd
import boto3
from io import StringIO
from json import JSONDecodeError
from typing import Tuple, Any, Optional, Literal

# Third-party libraries
from langchain_community.llms import Bedrock
import ipywidgets as widgets
from IPython.display import display, HTML

# --- Constants ---
AWS_REGION = "us-east-1"
MODEL_ID = "cohere.command-text-v14"
MODEL_KWARGS = {
    "max_tokens": 4096,
    "temperature": 0.0,
    "p": 0.01,
    "k": 0,
    "stop_sequences": [],
    "return_likelihoods": "NONE",
}

# --- Core Functions ---

def setup_aws_credentials():
    """
    Checks for and sets up AWS credentials.
    Best practice is to configure credentials via the AWS CLI (`aws configure`)
    or by setting environment variables, rather than hardcoding them.
    This function simply ensures the region is set.
    """
    if "AWS_SECRET_ACCESS_KEY" not in os.environ:
        print("Warning: AWS credentials not found in environment variables.")
        print("Please configure them using 'aws configure' or set environment variables.")
    os.environ["AWS_REGION"] = AWS_REGION
    # Ensure boto3 uses the region from the environment
    boto3.setup_default_session(region_name=AWS_REGION)


def create_bedrock_llm() -> Bedrock:
    """
    Creates and returns a Bedrock LLM instance with pre-defined settings.

    Returns:
        Bedrock: An instance of the LangChain Bedrock LLM.
    """
    llm = Bedrock(
        model_id=MODEL_ID,
        model_kwargs=MODEL_KWARGS
    )
    return llm

def parse_llm_response(
    response_text: str,
    output_format: Literal['JSON', 'CSV']
) -> Tuple[bool, Any, Optional[Exception]]:
    """
    Tries to parse the LLM's text response into the desired format (JSON or CSV).

    Args:
        response_text (str): The raw string output from the language model.
        output_format (Literal['JSON', 'CSV']): The target format.

    Returns:
        Tuple[bool, Any, Optional[Exception]]: A tuple containing:
            - has_error (bool): True if parsing failed, False otherwise.
            - content (Any): The parsed JSON object or Pandas DataFrame, or the raw text on error.
            - error (Optional[Exception]): The exception object if an error occurred, else None.
    """
    if output_format == 'JSON':
        try:
            # The model might wrap the JSON in ```json ... ```, so we find it.
            json_str_match = response_text[response_text.find('{'):response_text.rfind('}')+1]
            parsed_json = json.loads(json_str_match)
            return False, parsed_json, None
        except JSONDecodeError as e:
            return True, response_text, e
    elif output_format == 'CSV':
        try:
            # Clean the response to only include the CSV data
            lines = response_text.strip().split('\n')
            # Filter out potential markdown code fences or other text
            csv_lines = [line for line in lines if ',' in line and not line.startswith('```')]
            csv_data = "\n".join(csv_lines)
            
            if not csv_data:
                raise ValueError("No valid CSV data found in the response.")

            csv_io = StringIO(csv_data)
            df = pd.read_csv(csv_io)
            return False, df, None
        except Exception as e:
            return True, response_text, e
    return True, response_text, ValueError("Unsupported output format specified.")


def run_extraction(prompt: str, output_format: Literal['JSON', 'CSV']) -> Tuple[bool, Any, Optional[Exception]]:
    """
    Main extraction logic: invokes the LLM and parses the response.

    Args:
        prompt (str): The user-provided prompt for data extraction.
        output_format (Literal['JSON', 'CSV']): The desired output format.

    Returns:
        Tuple[bool, Any, Optional[Exception]]: The result from the parsing function.
    """
    llm = create_bedrock_llm()
    print(f"Invoking model for {output_format} extraction...")
    response_text = llm.invoke(prompt)
    print("Model invocation complete. Parsing response...")
    return parse_llm_response(response_text, output_format)


# --- UI Components and Event Handlers ---

# Define UI widgets
prompt_input = widgets.Textarea(
    value='Extract the name, age, and city from the following sentence:\n"John Doe is a 30-year-old software engineer who lives in New York."',
    placeholder='Enter your text and instructions here...',
    description='Prompt:',
    layout=widgets.Layout(width='90%', height='200px')
)

format_selector = widgets.RadioButtons(
    options=['JSON', 'CSV'],
    description='Output Format:',
    disabled=False
)

run_button = widgets.Button(
    description='Extract Data',
    button_style='primary',
    tooltip='Run the extraction process'
)

output_area = widgets.Output(layout=widgets.Layout(width='90%'))

def on_button_clicked(b):
    """
    Handles the click event of the run button.
    It clears the output, runs the extraction, and displays the result or an error.
    """
    with output_area:
        output_area.clear_output()
        display(HTML(value="<h3>Running...</h3>"))

        prompt_text = prompt_input.value
        selected_format = format_selector.value

        # Add a hint to the prompt to guide the model
        full_prompt = (
            f"{prompt_text}\n\n"
            f"Please provide the output strictly in {selected_format} format."
        )

        has_error, content, error = run_extraction(full_prompt, selected_format)

        output_area.clear_output()

        if not has_error:
            display(HTML(f"<h3>Result ({selected_format}):</h3>"))
            if selected_format == 'JSON':
                # Pretty print JSON
                json_output = f"<pre>{json.dumps(content, indent=2)}</pre>"
                display(HTML(json_output))
            elif selected_format == 'CSV':
                # Display DataFrame as an HTML table
                display(content)
        else:
            display(HTML(f"<h3 style='color:red;'>Failed to Parse {selected_format}</h3>"))
            display(HTML(f"<b>Error:</b> <pre>{error}</pre>"))
            display(HTML("<b>Raw Model Output:</b>"))
            display(HTML(f"<pre>{content}</pre>"))

def main():
    """
    Sets up credentials and displays the UI components.
    """
    setup_aws_credentials()
    run_button.on_click(on_button_clicked)

    # Display UI
    display(HTML(value="<h1>Bedrock Data Extractor</h1>"))
    display(HTML(value="<p>Enter text and specify the extraction task. Then choose the desired output format and run the model.</p>"))
    display(prompt_input)
    display(format_selector)
    display(run_button)
    display(output_area)


if __name__ == '__main__':
    # This block allows the script to be run in a Jupyter-like environment.
    # To run, open a Jupyter Notebook and use `%run app.py`.
    main()