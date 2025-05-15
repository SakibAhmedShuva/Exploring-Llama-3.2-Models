# Exploring Llama 3.2 and Other Models for JSON-based NER

This repository contains Jupyter notebooks and experiments focused on evaluating the capabilities of various Llama 3.2 models (from Hugging Face Transformers and GGUF formats) and Gemma models for Named Entity Recognition (NER). The primary goal is to extract structured information from text (specifically, driver's license-like data) and output it in a predefined JSON format, strictly adhering to a given system prompt.

## Project Overview

The core task is to instruct Language Models to:
1.  Act as a JSON-only response system.
2.  Follow a specific JSON schema for "address" or general NER requests.
3.  Adhere to strict rules like avoiding markdown, explanations, or date extraction unless explicitly part of the schema.

Two main notebooks are provided:
*   `Llama-3.2.ipynb`: Explores the `unsloth/Llama-3.2-1B-Instruct` model using the Hugging Face `transformers` library.
*   `Llama-3.2_gguf.ipynb`: Explores various GGUF (quantized) versions of Llama 3.2 (1B and 3B parameters) and Gemma 3 1B models using `llama-cpp-python`.

## Features

*   Demonstrates usage of Hugging Face `transformers` pipeline for text generation.
*   Demonstrates usage of `llama-cpp-python` for running GGUF models locally.
*   Tests various model sizes and quantizations (BF16, Q8_0, IQ4_XS, UD-IQ1_S, etc.).
*   Focuses on evaluating model adherence to complex system prompts and JSON output constraints.
*   Includes performance metrics (time taken for generation).

## Prerequisites

*   Python 3.8+
*   `pip` for installing packages

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SakibAhmedShuva/Exploring-Llama-3.2-Models.git
    cd Exploring-Llama-3.2-Models
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required Python packages:**
    A `requirements.txt` file should be created with the following content:
    ```
    transformers
    torch
    llama-cpp-python
    # Add any other specific versions if needed, e.g., accelerate
    ```
    Then install using:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `llama-cpp-python` might require compilation. If you encounter issues, refer to its official documentation for C/C++ compiler prerequisites for your OS.*

4.  **Jupyter Notebook Environment:**
    Ensure you have Jupyter Notebook or JupyterLab installed:
    ```bash
    pip install notebook jupyterlab
    ```

5.  **Model Downloads:**
    *   The Hugging Face `transformers` model (`unsloth/Llama-3.2-1B-Instruct`) will be downloaded automatically when the `pipeline` is initialized in `Llama-3.2.ipynb`.
    *   The GGUF models in `Llama-3.2_gguf.ipynb` will be downloaded automatically by `llama-cpp-python` from Hugging Face Hub upon first run of the respective cells. This might take some time depending on your internet connection and the model size.

## Usage

Open and run the Jupyter Notebooks:

1.  **`Llama-3.2.ipynb`**:
    *   This notebook uses the `unsloth/Llama-3.2-1B-Instruct` model via the `transformers` library.
    *   It defines a `system_prompt` and `test_instruction` to guide the model's JSON output.
    *   Run the cells sequentially to initialize the model and observe its output.

2.  **`Llama-3.2_gguf.ipynb`**:
    *   This notebook experiments with various GGUF models:
        *   `unsloth/Llama-3.2-1B-Instruct-GGUF` (Q8_0, IQ4_XS, BF16 quantizations)
        *   `unsloth/Llama-3.2-3B-Instruct-GGUF` (UD-IQ1_S, UD-IQ2_XXS, UD-IQ3_XXS, IQ4_XS quantizations)
        *   `unsloth/gemma-3-1b-it-GGUF` (BF16 quantization)
    *   Each model is loaded and tested against the same `system_prompt` and `test_instruction`.
    *   Run cells for the specific model you wish to test.

## Observed Behavior & Results (Summary)

The notebooks document the performance and adherence of different models to the specified JSON output format.

*   **`Llama-3.2.ipynb` (Hugging Face `unsloth/Llama-3.2-1B-Instruct`):**
    *   The model, when run with the provided parameters, did not strictly adhere to the JSON-only constraint or the `address` schema. It produced a more verbose, general NER JSON structure rather than the specific format requested in the system prompt.
    *   Execution Time: ~575 seconds (on Colab CPU).

*   **`Llama-3.2_gguf.ipynb` (GGUF Models):**
    *   **Llama-3.2-1B-Instruct-GGUF (Q8_0):** Adhered well to the `address` schema, extracting relevant fields.
        *   Execution Time: ~34 seconds.
    *   **Llama-3.2-1B-Instruct-GGUF (IQ4_XS):** Produced the placeholder JSON from the system prompt, indicating difficulty in following the extraction instruction with this quantization.
        *   Execution Time: ~35 seconds.
    *   **Llama-3.2-1B-Instruct-GGUF (BF16):** Adhered well to the `address` schema.
        *   Execution Time: ~61 seconds (slower than Q8_0).
    *   **Llama-3.2-3B-Instruct-GGUF (UD-IQ1_S):** Output consisted of repetitive "##" characters, failing the task.
        *   Execution Time: ~238 seconds.
    *   **Llama-3.2-3B-Instruct-GGUF (UD-IQ2_XXS):** Outputted a generic NER JSON structure, not the specified `address` schema.
        *   Execution Time: ~176 seconds.
    *   **Llama-3.2-3B-Instruct-GGUF (UD-IQ3_XXS):** Similar to UD-IQ2_XXS, outputted a generic NER JSON structure.
        *   Execution Time: ~238 seconds.
    *   **Llama-3.2-3B-Instruct-GGUF (IQ4_XS):** Produced a more structured NER JSON output but did not fully adhere to the `address` schema (e.g., nested `entities` key).
        *   Execution Time: ~153 seconds.
    *   **Gemma-3-1B-IT-GGUF (BF16):** Outputted the placeholder JSON and included markdown backticks, violating a system prompt constraint.
        *   Execution Time: ~52 seconds.

**General Observations:**
*   Smaller, well-quantized GGUF models (like Llama-3.2-1B Q8_0 and BF16) showed better adherence to the specific JSON schema for this task compared to the larger 3B GGUF models or the Hugging Face Transformers 1B model with the current setup.
*   The 3B GGUF models and the Hugging Face 1B model struggled significantly with either outputting a different JSON structure or failing to produce valid JSON as per the prompt.
*   The Gemma model also failed to follow the instructions accurately.
*   Performance varies significantly with model size and quantization.

## Potential Improvements & Future Work

*   **Prompt Engineering:** Further refine the system prompt and user instructions to improve adherence, especially for models that struggled.
*   **Parameter Tuning:** Experiment with different generation parameters (temperature, top_p, etc.) for both `transformers` and `llama-cpp-python`.
*   **Fine-tuning:** For highly specific and critical JSON extraction tasks, fine-tuning a base model on domain-specific data with the target JSON format would likely yield the best results.
*   **Error Handling & Validation:** Implement robust JSON parsing and validation against a Pydantic model or JSON schema.
*   **Broader Model Evaluation:** Test more models and quantization levels.
*   **Batch Processing:** For multiple inputs, explore batch inference capabilities.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
