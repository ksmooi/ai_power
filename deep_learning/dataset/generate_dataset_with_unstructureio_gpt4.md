
# Generating Datasets with Unstructured.IO and GPT-4

## Introduction
In the rapidly evolving field of artificial intelligence, the ability to efficiently process and understand unstructured data from various sources is crucial. Unstructured.IO offers powerful tools for extracting and organizing data from complex documents, such as PDFs. By leveraging the capabilities of OpenAI's GPT-4, we can dynamically generate system prompts and questions based on the content extracted from these documents, creating rich conversational datasets.

This article demonstrates how to integrate Unstructured.IO with GPT-4 to transform PDF documents into structured datasets. We will walk through the process of installing necessary libraries, extracting elements from PDFs, and using GPT-4's advanced language and vision capabilities to create dynamic and contextually relevant prompts and questions. These datasets can be used for training AI models, enhancing data analysis, and improving interactive applications.

By the end of this article, you will have a comprehensive understanding of how to utilize Unstructured.IO and GPT-4 to automate the creation of AI-ready datasets, enabling you to harness the full potential of your unstructured data.


## Prepare the `unstructured` Library
If you haven’t installed Docker on your machine, you can find the installation guide [here](https://docs.docker.com/get-docker/).

> **Note**: We build multi-platform images to support both x86\_64 and Apple silicon hardware. Using `docker pull` should download the appropriate image for your architecture. However, if needed, you can specify the platform with the `--platform` flag, e.g., `--platform linux/amd64`.

**1. Pulling the Docker Image**

We create Docker images for every push to the main branch. These images are tagged with the respective short commit hash (like `fbc7a69`) and the application version (e.g., `0.5.5-dev1`). The most recent image also receives the `latest` tag. To use these images, pull them from our repository:

```bash
docker pull downloads.unstructured.io/unstructured-io/unstructured:latest
```

**2. Using the Docker Image**

After pulling the image, you can create and start a container from it:

```bash
# create the container
docker run -dt --name unstructured downloads.unstructured.io/unstructured-io/unstructured:latest

# start a bash shell inside the running Docker container
docker exec -it unstructured bash
```

**3. Building Your Own Docker Image**

You can also build your own Docker image. If you only plan to parse a single type of data, you can accelerate the build process by excluding certain packages or requirements needed for other data types. Refer to the Dockerfile to determine which lines are necessary for your requirements.

```bash
make docker-build

# start a bash shell inside the running Docker container
make docker-start-bash
```

**4. Interacting with Python Inside the Container**

Once inside the running Docker container, you can directly test the library using Python’s interactive mode:

```python
python3

>>> from unstructured.partition.pdf import partition_pdf
>>> elements = partition_pdf(filename="example-docs/layout-parser-paper-fast.pdf")

>>> from unstructured.partition.text import partition_text
>>> elements = partition_text(filename="example-docs/fake-text.txt")
```


## Prepare the `openai` Library
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


## Example 1: Generating System Prompts and Questions from PDF Files
This example demonstrates how to process PDF files to generate a dataset of system prompts and questions dynamically using OpenAI's GPT-4-turbo model. The script partitions the PDF files to extract elements, generates unique IDs, and uses the content of each element to dynamically create a system prompt and a relevant question. The results are compiled into a dataset and saved as a CSV file.

### Example Code
```python
# Import necessary libraries
import os
import uuid
import openai
from unstructured.partition.pdf import partition_pdf
import pandas as pd

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set directly like openai.api_key = "your-api-key"

def generate_system_prompt_and_question(text):
    """
    Generates a system prompt and a relevant question based on the provided text content
    using OpenAI's GPT-4-turbo model.

    Args:
        text (str): The text content to base the prompt and question on.

    Returns:
        tuple: A tuple containing the system prompt and the question.
    """
    # Make a call to OpenAI to generate system prompt and question
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that generates system prompts and questions based on given text content."
            },
            {
                "role": "user",
                "content": f"Given the following content, generate a system prompt and a relevant question: {text}"
            }
        ]
    )

    # Parse the response to extract the system prompt and question
    messages = response['choices'][0]['message']['content'].strip().split('\n')
    system_prompt = messages[0].replace("System Prompt:", "").strip()
    question = messages[1].replace("Question:", "").strip()
    return system_prompt, question

def process_pdf_to_dataset(pdf_files):
    """
    Processes a list of PDF files to generate a dataset of system prompts and questions.

    Args:
        pdf_files (list): List of paths to PDF files.

    Returns:
        pd.DataFrame: A DataFrame containing the dataset.
    """
    data = []

    for pdf_file in pdf_files:
        # Partition the PDF file to retrieve elements
        elements = partition_pdf(filename=pdf_file, strategy="auto")

        for element in elements:
            # Generate a unique ID for each entry
            element_id = str(uuid.uuid4())

            # Generate system prompt and question dynamically
            system_prompt, question = generate_system_prompt_and_question(element.text)

            # Append the data to the list
            data.append({
                'id': element_id,
                'system_prompt': system_prompt,
                'question': question,
                'response': element.text
            })

    # Create a DataFrame from the list of data
    df = pd.DataFrame(data)
    return df

# List of paths to PDF files to be processed
pdf_files = [
    "path/to/your/first.pdf",
    "path/to/your/second.pdf",
    "path/to/your/third.pdf"
]

# Process the PDF files to create the dataset
dataset = process_pdf_to_dataset(pdf_files)

# Save the dataset to a CSV file
dataset.to_csv("extracted_content_dataset.csv", index=False)
```

### Explanation

1. **Import Necessary Libraries**:
   - Import libraries for file handling, UUID generation, OpenAI API interaction, PDF partitioning, and data manipulation.
2. **Set Up OpenAI API Key**:
   - Set up the OpenAI API key to authenticate API requests.
3. **Generate System Prompt and Question Function**:
   - Define a function that uses OpenAI's GPT-4-turbo model to generate a system prompt and a relevant question based on the provided text content.
4. **Process PDF to Dataset Function**:
   - Define a function that processes a list of PDF files:
     - Partition each PDF file to extract elements.
     - For each element, generate a unique ID and use the `generate_system_prompt_and_question` function to create prompts and questions.
     - Append the generated data to a list.
     - Convert the list to a pandas DataFrame and return it.
5. **Main Script**:
   - Define a list of paths to PDF files to be processed.
   - Call the `process_pdf_to_dataset` function to create the dataset.
   - Save the dataset to a CSV file.


## Example 2: Extracting Elements from Microsoft Office Files and Text Files
Here is a complete example demonstrating how to create a dataset by retrieving content from Microsoft Word, Excel, PowerPoint, and text files, extracting elements, and generating dynamic system prompts and questions using OpenAI's GPT-4-turbo. The script will partition the documents to identify elements, generate unique IDs, and use the content of each element to dynamically create system prompts and relevant questions. The results are compiled into a dataset, including image references and conversation data, and saved as a CSV file.

### Example Code
```python
import os
import uuid
import openai
import pandas as pd
from unstructured.partition.doc import partition_doc
from unstructured.partition.docx import partition_docx
from unstructured.partition.xlsx import partition_xlsx
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.text import partition_text

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set directly like openai.api_key = "your-api-key"

def generate_system_prompt_and_question(text):
    """
    Generates a system prompt and a relevant question based on the given text content.
    
    Args:
        text (str): The text content to generate the prompt and question for.
    
    Returns:
        tuple: A tuple containing the generated system prompt and question.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that generates system prompts and questions based on given text content."
            },
            {
                "role": "user",
                "content": f"Given the following content, generate a system prompt and a relevant question: {text}"
            }
        ]
    )

    # Split the response content into system prompt and question
    messages = response['choices'][0]['message']['content'].strip().split('\n')
    system_prompt = messages[0].replace("System Prompt:", "").strip()
    question = messages[1].replace("Question:", "").strip()
    return system_prompt, question

def process_documents_to_dataset(files):
    """
    Processes a list of document files to create a dataset by extracting elements and generating dynamic system prompts and questions.
    
    Args:
        files (list): A list of file paths to process.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the processed data with unique IDs, file names, text, system prompts, and questions.
    """
    data = []

    for file in files:
        # Determine the file extension and partition the document accordingly
        file_extension = os.path.splitext(file)[1].lower()
        if file_extension == '.doc':
            elements = partition_doc(filename=file)
        elif file_extension == '.docx':
            elements = partition_docx(filename=file)
        elif file_extension == '.xlsx':
            elements = partition_xlsx(filename=file)
        elif file_extension == '.pptx':
            elements = partition_pptx(filename=file)
        elif file_extension in ['.txt', '.text']:
            elements = partition_text(filename=file)
        else:
            continue

        # Process each element extracted from the document
        for element in elements:
            # Generate a unique ID for each entry
            element_id = str(uuid.uuid4())

            # Generate system prompt and question dynamically based on the element text
            system_prompt, question = generate_system_prompt_and_question(element.text)

            # Append the data to the list
            data.append({
                'id': element_id,
                'file_name': os.path.basename(file),
                'text': element.text,
                'system_prompt': system_prompt,
                'question': question
            })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    return df

# List of files to process
files = [
    "path/to/your/document1.doc",
    "path/to/your/document2.docx",
    "path/to/your/spreadsheet.xlsx",
    "path/to/your/presentation.pptx",
    "path/to/your/textfile.txt"
]

# Process the documents and create the dataset
dataset = process_documents_to_dataset(files)

# Save the dataset to a CSV file
dataset.to_csv("generated_dataset.csv", index=False)
```

### Explanation

1. **Install Required Libraries**:
   We install the necessary libraries for processing different document types and for interacting with the OpenAI API.
2. **Set Up OpenAI API Key**:
   We ensure that the OpenAI API key is set up properly.
3. **Define Functions**:
   - `generate_system_prompt_and_question(text)`: This function takes text as input and uses GPT-4-turbo to generate a system prompt and a relevant question.
   - `process_documents_to_dataset(files)`: This function processes a list of document files (Word, Excel, PowerPoint, and text), extracts elements, and generates dynamic system prompts and questions for each element. The results are compiled into a dataset.
4. **Process Documents**:
   We specify the paths to the document files to be processed and call the `process_documents_to_dataset` function to generate the dataset.
5. **Save the Dataset**:
   The dataset is saved to a CSV file for further use.


## Example 3: Processing Image Elements with Vision Capabilities
This example demonstrates how to process image elements from PDFs and generate conversational prompts and questions using OpenAI's GPT-4 with vision capabilities. The script includes functionality to check for image elements, save image data to files, and generate system prompts and questions based on the image content. By handling only image elements, the script ensures relevant and accurate data processing, resulting in a conversational dataset format that leverages advanced AI capabilities for enhanced interaction and understanding of image-based content.

### Example Code
```python
import os
import uuid
import openai
from unstructured.partition.pdf import partition_pdf
import pandas as pd

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")  # or set directly like openai.api_key = "your-api-key"

def generate_system_prompt_and_question(image_path):
    """
    Generates a system prompt and a relevant question based on the given image content.
    
    Args:
        image_path (str): The file path to the image.
    
    Returns:
        tuple: A tuple containing the generated system prompt and question.
    """
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that generates system prompts and questions based on given image content."
            },
            {
                "role": "user",
                "content": "Generate a system prompt and a relevant question based on the following image."
            }
        ],
        files=[
            {
                "file": image_data,
                "filename": image_path
            }
        ]
    )

    messages = response['choices'][0]['message']['content'].strip().split('\n')
    system_prompt = messages[0].replace("System Prompt:", "").strip()
    question = messages[1].replace("Question:", "").strip()
    return system_prompt, question

def process_pdf_to_dataset(pdf_files):
    """
    Processes a list of PDF files to create a dataset by extracting image elements and generating dynamic system prompts and questions.
    
    Args:
        pdf_files (list): A list of PDF file paths to process.
    
    Returns:
        pandas.DataFrame: A DataFrame containing the processed data with unique IDs, image paths, and conversations.
    """
    data = []

    for pdf_file in pdf_files:
        # Partition the PDF file
        elements = partition_pdf(filename=pdf_file, strategy="auto")

        for element in elements:
            if element.type == "Image" and 'image' in element.metadata:
                # Generate a unique ID for each entry
                element_id = str(uuid.uuid4())

                # Retrieve the image file name and save the image data to a file
                image_data = element.metadata['image']
                image_filename = f"{os.path.basename(pdf_file).replace('.pdf', '')}_{element_id}.jpg"
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_data)

                # Generate system prompt and question dynamically
                system_prompt, question = generate_system_prompt_and_question(image_filename)

                # Append the data to the list
                data.append({
                    'id': element_id,
                    'image': image_filename,
                    'conversations': [
                        {"from": "human", "value": f"<image>\n{question}"},
                        {"from": "gpt", "value": element.text}
                    ]
                })

    # Create a DataFrame
    df = pd.DataFrame(data)
    return df

# List of PDF files to process
pdf_files = [
    "path/to/your/first.pdf",
    "path/to/your/second.pdf",
    "path/to/your/third.pdf"
]

# Process the PDF files and create the dataset
dataset = process_pdf_to_dataset(pdf_files)

# Save the dataset to a CSV file
dataset.to_csv("extracted_content_dataset.csv", index=False)
```

### Explanation

1. **Generate Prompts and Questions Based on Image**:
   - The `generate_system_prompt_and_question` function reads the image data and sends it to the OpenAI API with vision capabilities to generate prompts and questions based on the image content.
2. **Check for Image Type and Metadata**:
   - The script checks if the `element.type` is `"Image"` and if `'image'` is in `element.metadata` to ensure we only process valid image elements.
3. **Save Image Data**:
   - The script saves the image data from the metadata to a file and uses this file to generate the system prompt and question.
4. **Generate System Prompts and Questions**:
   - For each valid image element, the script generates a unique ID, retrieves the image data, saves it to a file, creates the dynamic system prompt and question using the image file, and appends the relevant data to the dataset.


## Conclusion
In this article, we demonstrated how to create a dataset by retrieving content from various document types, including Microsoft Word, Excel, PowerPoint, and text files. By leveraging Unstructured.IO's powerful partitioning capabilities and OpenAI's GPT-4-turbo, we were able to extract elements from these documents and generate dynamic system prompts and relevant questions. This approach ensures that the dataset is rich, contextually relevant, and ready for various AI applications.

The provided script partitions the documents, identifies elements, generates unique IDs, and uses the content of each element to dynamically create prompts and questions. The results are compiled into a structured dataset, including image references and conversation data, which is then saved as a CSV file. This process not only automates the dataset creation but also enhances the interaction and understanding of document-based content through advanced AI capabilities.

By following the steps outlined, you can create customized conversational datasets tailored to your specific needs, enabling more effective training of AI models and improving data analysis workflows. 


## References
- [Unstructured.IO Documentation](https://github.com/Unstructured-IO/unstructured)
- [OpenAI Python Library Documentation](https://platform.openai.com/docs/libraries/python-library)

