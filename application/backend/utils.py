import os
import pathlib
from google import genai
from google.genai import types


def get_root_folder_path():
    path = os.getcwd()
    path_clean = []
    for x in path.split('/'):
        if x != 'ecir-2026-demo':
            path_clean.append(x)
        else:
            path_clean.append(x)
            break
    return '/'.join(path_clean)


def load_api_key(model_type='google'):
    """Load API key from environment variable or file.
    
    Priority:
    1. Environment variable (GEMINI_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY)
    2. File (<model_type>_api_key.txt)
    
    Args:
        model_type: Type of model ('google', 'openai', 'anthropic')
        
    Returns:
        API key string
    """
    assert model_type in ['google', 'openai', 'anthropic']
    
    # Map model types to environment variable names
    env_var_map = {
        'google': ['GEMINI_API_KEY', 'GOOGLE_API_KEY'],  # Try both
        'openai': ['OPENAI_API_KEY'],
        'anthropic': ['ANTHROPIC_API_KEY']
    }
    
    # Try to get from environment variables first
    for env_var in env_var_map[model_type]:
        api_key = os.environ.get(env_var)
        if api_key:
            return api_key.strip()
    
    # Fall back to file-based API key (for local development)
    try:
        path_to_key = f'{get_root_folder_path()}/{model_type}_api_key.txt'
        with open(path_to_key, 'r') as key_file:
            key = key_file.read()
        return key.strip()
    except FileNotFoundError:
        raise ValueError(
            f"API key not found for {model_type}. "
            f"Set environment variable {env_var_map[model_type][0]} "
            f"or create file {model_type}_api_key.txt"
        )


def load_sample_pdf_files():
    pdf_files = {}
    for file in os.listdir(f'{get_root_folder_path()}/backend/input_files'):
        if file.endswith('pdf'):
            pdf_files[file] = pathlib.Path(f'{get_root_folder_path()}/backend/input_files/{file}').read_bytes()
    return pdf_files
