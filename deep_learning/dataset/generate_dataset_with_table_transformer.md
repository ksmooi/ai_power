# Generating Datasets with Table Transformer and GPT-4

## Introduction
The advancements in AI have significantly enhanced the capabilities for data extraction and processing, particularly in the realm of document analysis. The Table Transformer and OpenAI's GPT-4 are two powerful tools that can be leveraged to extract, recognize, and generate insightful data from structured documents such as PDFs. The Table Transformer, developed by Microsoft, is designed for detecting and extracting table structures from documents, while GPT-4 offers advanced language understanding and generation capabilities.

This article provides a comprehensive guide on generating datasets using these technologies. We will explore several examples, including detecting tables in images, extracting tables from PDFs, and recognizing table structures in images. Additionally, we will demonstrate how to integrate GPT-4 for dynamically generating prompts and questions based on the extracted content.


## Prepare the `Table Transformer` library
To set up your environment for working with the code, follow these steps:

1. **Create a Conda Environment**:
   - Use the provided `environment.yml` file to create a new Conda environment by running the following command:
     ```bash
     conda env create -f environment.yml
     ```
2. **Activate the Environment**:
   - Once the environment is created, activate it using the command:
     ```bash
     conda activate tables-detr
     ```

This will set up all the necessary dependencies and activate the environment for your project.


## Prepare the `openai` library
We provide a Python library, which you can install by running:

```bash
pip install openai
```

Once installed, you can use the library and your secret key to run the following:

```python
from openai import OpenAI

client = OpenAI(
    # Defaults to os.environ.get("OPENAI_API_KEY")
)

chat_completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello world"}]
)
```

The bindings also will install a command-line utility you can use as follows:

```bash
$ openai api chat_completions.create -m gpt-3.5-turbo -g user "Hello world"
```


## Example 1: Detecting Tables in Images
This example demonstrates how to use the `TableExtractionPipeline` class's `detect` method to identify and extract tables from an image. The process includes loading an image and its associated text tokens, calling the `detect` method with parameters to specify the desired outputs (detected objects and cropped tables), and handling the results by printing the detected objects and saving the cropped table images and their associated tokens to files. This example provides a comprehensive guide on using the `detect` method to process images and extract table-related information using the specified models and configurations.

### Example Code
```python
# Import necessary libraries
import json
from PIL import Image
import torch

# Assuming you have already defined the TableExtractionPipeline class
class TableExtractionPipeline:
    # The class definition as provided earlier
    pass

# Instantiate the pipeline with the necessary configurations and model paths
pipeline = TableExtractionPipeline(
    det_device='cuda', 
    str_device='cuda',
    det_config_path='path/to/det_config.json', 
    det_model_path='path/to/det_model.pth',
    str_config_path='path/to/str_config.json', 
    str_model_path='path/to/str_model.pth'
)

# Load an image and tokens
img = Image.open('path/to/image.jpg')
tokens = json.load(open('path/to/tokens.json', 'r'))

# Ensure the tokens are in the correct format (e.g., list of dictionaries with 'text' and 'bbox' keys)
# This is a simple example format for tokens
tokens = [
    {"text": "example", "bbox": [50, 50, 100, 100], "line_num": 0, "block_num": 0, "span_num": 0},
    # Add more tokens as needed
]

# Call the detect method with all parameters
result = pipeline.detect(
    img,
    tokens=tokens,
    out_objects=True,
    out_crops=True,
    crop_padding=10
)

# Handle the result
# The result is a dictionary with keys 'objects' and 'crops'
print("Detection Results:")

# Extract and print detected objects
objects = result.get('objects', [])
print("Detected Objects:")
for obj in objects:
    print(obj)

# Save cropped table images and tokens
crops = result.get('crops', [])
for crop_idx, crop in enumerate(crops):
    # Save the cropped image
    crop_img_path = f'path/to/output_crop_{crop_idx + 1}.jpg'
    crop['image'].save(crop_img_path)

    # Save the tokens for the cropped image
    crop_tokens_path = f'path/to/output_crop_{crop_idx + 1}_tokens.json'
    with open(crop_tokens_path, 'w') as tokens_file:
        json.dump(crop['tokens'], tokens_file)

    print(f"Cropped Image {crop_idx + 1} saved with tokens.")
```

### Explanation

1. **Pipeline Instantiation**:
    - The `TableExtractionPipeline` class is instantiated with the required device settings and model configuration paths.
2. **Loading Image and Tokens**:
    - The input image is loaded using `PIL.Image.open`.
    - Tokens are loaded from a JSON file and structured correctly as a list of dictionaries. Each dictionary contains the text, bounding box, and other optional attributes for each token.
3. **Calling `detect` Method**:
    - The `detect` method of the pipeline is called with all parameters:
        - `img`: The input image.
        - `tokens`: List of tokens.
        - `out_objects`: Whether to output detected objects.
        - `out_crops`: Whether to output cropped tables.
        - `crop_padding`: Padding around the cropped tables.
4. **Handling the Result**:
    - The result from the `detect` method is a dictionary with keys `objects` and `crops`.
    - **Detected Objects**: Print the detected objects.
    - **Cropped Tables**: Save each cropped table image and its associated tokens.
        - **Cropped Image**: Save the cropped image to a file.
        - **Cropped Tokens**: Save the tokens for the cropped image to a JSON file.


## Example 2: Extracting Tables from PDFs
This example demonstrates how to use the `PDFTableExtractor` class to extract tables from a PDF file. The process involves partitioning the PDF to identify table elements, converting specific pages to images, performing OCR to extract text tokens, and utilizing the `TableExtractionPipeline` to detect and recognize table structures. The results, including detected objects, table cells, HTML, CSV, and cropped table images, are saved in a specified output directory. This example showcases the complete workflow from PDF to structured table extraction and output.

### Example Code
```python
# Import necessary libraries
import json
from PIL import Image
import torch

# Assuming you have already defined the TableExtractionPipeline class
class TableExtractionPipeline:
    # The class definition as provided earlier
    pass

# Instantiate the pipeline with the necessary configurations and model paths
pipeline = TableExtractionPipeline(
    det_device='cuda',            # Device for the detection model (e.g., 'cuda' for GPU)
    str_device='cuda',            # Device for the structure model (e.g., 'cuda' for GPU)
    det_config_path='path/to/det_config.json',  # Path to the detection model config file
    det_model_path='path/to/det_model.pth',     # Path to the detection model weights
    str_config_path='path/to/str_config.json',  # Path to the structure model config file
    str_model_path='path/to/str_model.pth'      # Path to the structure model weights
)

# Load an image and tokens
img = Image.open('path/to/image.jpg')
tokens = json.load(open('path/to/tokens.json', 'r'))

# Ensure the tokens are in the correct format (e.g., list of dictionaries with 'text' and 'bbox' keys)
# This is a simple example format for tokens
tokens = [
    {"text": "example", "bbox": [50, 50, 100, 100], "line_num": 0, "block_num": 0, "span_num": 0},
    # Add more tokens as needed
]

# Call the extract method of the TableExtractionPipeline with all parameters
result = pipeline.extract(
    img,              # The input image to be processed
    tokens=tokens,    # List of tokens extracted from the image using OCR
    out_objects=True, # Whether to output detected objects (e.g., table boundaries)
    out_crops=True,   # Whether to output cropped tables (sub-images of detected tables)
    out_cells=True,   # Whether to output detected cells (individual table cells)
    out_html=True,    # Whether to output the table in HTML format
    out_csv=True,     # Whether to output the table in CSV format
    crop_padding=10   # Padding around the cropped tables
)


# Handle the result
# The result is a list of extracted tables with their structures and content
for table_idx, extracted_table in enumerate(result):
    print(f"Table {table_idx + 1}:")
    
    # Extract and print detected objects
    objects = extracted_table.get('objects', [])
    print("Detected Objects:")
    for obj in objects:
        print(obj)

    # Extract and print detected cells
    cells = extracted_table.get('cells', [])
    print("\nDetected Cells:")
    for cell in cells:
        print(cell)

    # Save HTML output to a file
    html_output = extracted_table.get('html', [])
    if html_output:
        html_file_path = f'path/to/output_table_{table_idx + 1}.html'
        with open(html_file_path, 'w') as html_file:
            html_file.write(html_output[0])  # If there are multiple tables, handle accordingly

    # Save CSV output to a file
    csv_output = extracted_table.get('csv', [])
    if csv_output:
        csv_file_path = f'path/to/output_table_{table_idx + 1}.csv'
        with open(csv_file_path, 'w') as csv_file:
            csv_file.write(csv_output[0])  # If there are multiple tables, handle accordingly

    # Save cropped table images
    crops = extracted_table.get('crops', [])
    for crop_idx, crop in enumerate(crops):
        crop_img_path = f'path/to/output_table_{table_idx + 1}_crop_{crop_idx + 1}.jpg'
        crop['image'].save(crop_img_path)

        # Save tokens for each cropped table
        crop_tokens_path = f'path/to/output_table_{table_idx + 1}_crop_{crop_idx + 1}_tokens.json'
        with open(crop_tokens_path, 'w') as tokens_file:
            json.dump(crop['tokens'], tokens_file)

```

### Explanation

1. **Pipeline Instantiation**:
    - The `TableExtractionPipeline` class is instantiated with the required device settings and model configuration paths.
2. **Loading Image and Tokens**:
    - The input image is loaded using `PIL.Image.open`.
    - Tokens are loaded from a JSON file and structured correctly as a list of dictionaries. Each dictionary contains the text, bounding box, and other optional attributes for each token.
3. **Calling `extract` Method**:
    - The `extract` method of the pipeline is called with all parameters:
        - `img`: The input image.
        - `tokens`: List of tokens.
        - `out_objects`: Whether to output detected objects.
        - `out_crops`: Whether to output cropped tables.
        - `out_cells`: Whether to output detected cells.
        - `out_html`: Whether to output HTML.
        - `out_csv`: Whether to output CSV.
        - `crop_padding`: Padding around the cropped tables.
4. **Handling the Result**:
    - The result from the `extract` method is a list of extracted tables. Each table in the list contains detected objects, cells, HTML, CSV, and cropped images.
    - **Detected Objects**: Print the detected objects.
    - **Detected Cells**: Print the detected cells.
    - **HTML Output**: Save the HTML output to a file.
    - **CSV Output**: Save the CSV output to a file.
    - **Cropped Images and Tokens**: Save each cropped table image and its associated tokens.


## Example 3: Recognizing Table Structures in Images
This example demonstrates how to use the `TableExtractionPipeline` class's `recognize` method to extract and recognize table structures from an image. The process involves loading an image and its associated text tokens, calling the `recognize` method with parameters to specify the desired outputs (detected objects, cells, HTML, and CSV), and handling the results by printing detected objects and cells, and saving the HTML and CSV outputs to files. This example showcases the complete workflow for recognizing and extracting table information from an image using the specified models and configurations.

### Example Code
```python
# Import necessary libraries
import json
from PIL import Image
import torch

# Assuming you have already defined the TableExtractionPipeline class
class TableExtractionPipeline:
    # The class definition as provided earlier
    pass

# Instantiate the pipeline with the necessary configurations and model paths
pipeline = TableExtractionPipeline(
    det_device='cuda',            # Device for the detection model (e.g., 'cuda' for GPU)
    str_device='cuda',            # Device for the structure model (e.g., 'cuda' for GPU)
    det_config_path='path/to/det_config.json',  # Path to the detection model config file
    det_model_path='path/to/det_model.pth',     # Path to the detection model weights
    str_config_path='path/to/str_config.json',  # Path to the structure model config file
    str_model_path='path/to/str_model.pth'      # Path to the structure model weights
)


# Load an image and tokens
img = Image.open('path/to/image.jpg')
tokens = json.load(open('path/to/tokens.json', 'r'))

# Ensure the tokens are in the correct format (e.g., list of dictionaries with 'text' and 'bbox' keys)
# This is a simple example format for tokens
tokens = [
    {"text": "example", "bbox": [50, 50, 100, 100], "line_num": 0, "block_num": 0, "span_num": 0},
    # Add more tokens as needed
]

# Call the recognize method of the TableExtractionPipeline with all parameters
result = pipeline.recognize(
    img,                # The input image to be processed
    tokens=tokens,      # List of tokens extracted from the image using OCR
    out_objects=True,   # Whether to output detected objects (e.g., table boundaries)
    out_cells=True,     # Whether to output detected cells (individual table cells)
    out_html=True,      # Whether to output the table in HTML format
    out_csv=True        # Whether to output the table in CSV format
)

# Handle the result
# The result is a dictionary with keys 'objects', 'cells', 'html', 'csv'
# Print and save the outputs
print("Recognition Results:")

# Extract and print detected objects
objects = result.get('objects', [])
print("Detected Objects:")
for obj in objects:
    print(obj)

# Extract and print detected cells
cells = result.get('cells', [])
print("\nDetected Cells:")
for cell in cells:
    print(cell)

# Save HTML output to a file
html_output = result.get('html', [])
if html_output:
    html_file_path = 'path/to/output.html'
    with open(html_file_path, 'w') as html_file:
        html_file.write(html_output[0])  # If there are multiple tables, handle accordingly

# Save CSV output to a file
csv_output = result.get('csv', [])
if csv_output:
    csv_file_path = 'path/to/output.csv'
    with open(csv_file_path, 'w') as csv_file:
        csv_file.write(csv_output[0])  # If there are multiple tables, handle accordingly
```

### Explanation

1. **Pipeline Instantiation**:
    - The `TableExtractionPipeline` class is instantiated with the required device settings and model configuration paths.
2. **Loading Image and Tokens**:
    - The input image is loaded using `PIL.Image.open`.
    - Tokens are loaded from a JSON file and structured correctly as a list of dictionaries. Each dictionary contains the text, bounding box, and other optional attributes for each token.
3. **Calling `recognize` Method**:
    - The `recognize` method of the pipeline is called with all parameters:
        - `img`: The input image.
        - `tokens`: List of tokens.
        - `out_objects`: Whether to output detected objects.
        - `out_cells`: Whether to output detected cells.
        - `out_html`: Whether to output HTML.
        - `out_csv`: Whether to output CSV.
4. **Handling the Result**:
    - The result from the `recognize` method is a dictionary with keys `objects`, `cells`, `html`, and `csv`.
    - **Detected Objects**: Print the detected objects.
    - **Detected Cells**: Print the detected cells.
    - **HTML Output**: Save the HTML output to a file.
    - **CSV Output**: Save the CSV output to a file.


## Example 4: Comprehensive Table Extraction from PDFs
This example demonstrates how to extract tables from a PDF file using the PDFTableExtractor class, which leverages a detection and structure recognition pipeline (TableExtractionPipeline). The process involves partitioning the PDF to identify table elements, converting specific pages to images, performing OCR to extract tokens, and using the pipeline to detect and extract tables, which are then saved in various formats (JSON, HTML, CSV) in the specified output directory.

### Example Code
```python
import os
import json
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import torch
from partition import partition_pdf  # Assuming you have a partition_pdf function available

class PDFTableExtractor:
    """
    A class to extract tables from PDF files using a detection and structure recognition pipeline.
    """

    def __init__(self, det_device, str_device, det_config_path, det_model_path, str_config_path, str_model_path):
        """
        Initializes the PDFTableExtractor with specified devices and model paths/configs.

        Args:
            det_device (str): Device for detection model (e.g., 'cuda' or 'cpu').
            str_device (str): Device for structure model (e.g., 'cuda' or 'cpu').
            det_config_path (str): Path to the detection model config file.
            det_model_path (str): Path to the detection model weights.
            str_config_path (str): Path to the structure model config file.
            str_model_path (str): Path to the structure model weights.
        """     
        # Instantiate the pipeline with the necessary configurations and model paths
        self.pipeline = TableExtractionPipeline(
            det_device='cuda',            # Device for the detection model (e.g., 'cuda' for GPU)
            str_device='cuda',            # Device for the structure model (e.g., 'cuda' for GPU)
            det_config_path='path/to/det_config.json',  # Path to the detection model config file
            det_model_path='path/to/det_model.pth',     # Path to the detection model weights
            str_config_path='path/to/str_config.json',  # Path to the structure model config file
            str_model_path='path/to/str_model.pth'      # Path to the structure model weights
        )

    def extract_tables_from_pdf(self, pdf_path, output_dir):
        """
        Extracts tables from a PDF file and saves the results in the specified output directory.

        Args:
            pdf_path (str): Path to the input PDF file.
            output_dir (str): Path to the output directory.
        """
        # Partition PDF to retrieve elements
        elements = partition_pdf(pdf_path)
        
        for element in elements:
            if element['type'] == 'table':
                page_num = element['page_num']
                # Convert the specific page to an image
                img = convert_from_path(pdf_path, 300, first_page=page_num, last_page=page_num)[0]

                # Perform OCR (Optical Character Recognition) on the image to get text tokens
                # Initialize an empty list to store the tokens
                ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                tokens = []
                
                # Iterate over each detected element in the OCR data
                for i in range(len(ocr_data['level'])):
                    # Create a dictionary for each token with its text, bounding box, and additional metadata
                    token = {
                        'text': ocr_data['text'][i],                    # Extracted text
                        'bbox': [                                       # Bounding box coordinates
                            ocr_data['left'][i],                        # Left coordinate
                            ocr_data['top'][i],                         # Top coordinate
                            ocr_data['left'][i] + ocr_data['width'][i], # Right coordinate (left + width)
                            ocr_data['top'][i] + ocr_data['height'][i]  # Bottom coordinate (top + height)
                        ],
                        'line_num': ocr_data['line_num'][i],            # Line number of the text
                        'block_num': ocr_data['block_num'][i],          # Block number of the text
                        'span_num': i                                   # Span number (index) of the text
                    }
                    # Append the token dictionary to the tokens list
                    tokens.append(token)
                    
                # Extract tables using the pipeline
                extracted_tables = self.pipeline.extract(img, tokens, out_objects=True, out_cells=True, out_html=True, out_csv=True)
                
                # Save the results
                page_output_dir = os.path.join(output_dir, f'page_{page_num}')
                os.makedirs(page_output_dir, exist_ok=True)
                
                for table_idx, table in enumerate(extracted_tables):
                    # Save objects
                    objects_path = os.path.join(page_output_dir, f'table_{table_idx + 1}_objects.json')
                    with open(objects_path, 'w') as f:
                        json.dump(table.get('objects', []), f)
                    
                    # Save cells
                    cells_path = os.path.join(page_output_dir, f'table_{table_idx + 1}_cells.json')
                    with open(cells_path, 'w') as f:
                        json.dump(table.get('cells', []), f)
                    
                    # Save HTML
                    if table.get('html'):
                        html_path = os.path.join(page_output_dir, f'table_{table_idx + 1}.html')
                        with open(html_path, 'w') as f:
                            f.write(table['html'][0])  # If there are multiple tables, handle accordingly
                    
                    # Save CSV
                    if table.get('csv'):
                        csv_path = os.path.join(page_output_dir, f'table_{table_idx + 1}.csv')
                        with open(csv_path, 'w') as f:
                            f.write(table['csv'][0])  # If there are multiple tables, handle accordingly


def main():
    """
    Main function to run the PDFTableExtractor on a given PDF file.
    """
    # Example paths (replace with actual paths)
    pdf_path = 'path/to/your/file.pdf'
    output_dir = 'path/to/output/directory'
    
    # Instantiate the PDFTableExtractor with the necessary configurations and model paths
    extractor = PDFTableExtractor(
        det_device='cuda',                  # Device for the detection model (e.g., 'cuda' for GPU)
        str_device='cuda',                  # Device for the structure model (e.g., 'cuda' for GPU)
        det_config_path='path/to/det_config.json',  # Path to the detection model config file
        det_model_path='path/to/det_model.pth',     # Path to the detection model weights
        str_config_path='path/to/str_config.json',  # Path to the structure model config file
        str_model_path='path/to/str_model.pth'      # Path to the structure model weights
    )

    # Extract tables from the PDF
    extractor.extract_tables_from_pdf(pdf_path, output_dir)


if __name__ == "__main__":
    main()
```

### Explanation

1. **PDFTableExtractor Class**: This class encapsulates the functionality to extract tables from a PDF file.
    - **__init__**: Initializes the pipeline with specified model paths and devices.
    - **extract_tables_from_pdf**: Extracts tables from a given PDF file and saves the results.
2. **extract_tables_from_pdf Method**:
    - Uses `partition_pdf` to partition the PDF and retrieve elements.
    - Checks if the element type is 'table'.
    - Converts the specific page containing the table to an image.
    - Performs OCR on the image to get text tokens using `pytesseract`.
    - Calls the `extract` method of the pipeline to extract tables.
    - Saves the extracted tables in JSON, HTML, and CSV formats in the specified output directory.
3. **main Function**:
    - Instantiates the `PDFTableExtractor` with appropriate model paths and devices.
    - Calls the `extract_tables_from_pdf` method to extract tables from the given PDF file.


## Conclusion
Leveraging the Table Transformer and GPT-4 for dataset generation provides a robust and efficient method for extracting structured data from documents. By combining these technologies, we can automate the process of detecting and recognizing tables, converting them into various formats such as JSON, HTML, and CSV. This automation not only saves time but also enhances the accuracy and consistency of data extraction.

The examples provided in this article showcase the versatility and power of using Table Transformer and GPT-4 in various scenarios. From extracting tables in images to processing entire PDFs, these tools offer scalable solutions for handling large volumes of documents. As AI continues to evolve, the integration of such advanced models will further streamline data processing tasks, making them more accessible and efficient for a wide range of applications.


## References
- [Microsoft Table Transformer](https://github.com/microsoft/table-transformer)
- [OpenAI Python Library Documentation](https://platform.openai.com/docs/libraries/python-library)

