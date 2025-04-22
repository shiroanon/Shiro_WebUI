# --- START OF FILE base_utils.py ---

import utils.ShiroScript.parse_shiro as shiro
import random
import string
import requests
import re
import math
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Constants ---
DEFAULT_RETRY_COUNT = 3
DEFAULT_BACKOFF_FACTOR = 0.5 # Affects sleep time: {backoff factor} * (2 ** ({number of total retries} - 1))
DEFAULT_STATUS_FORCELIST = (500, 502, 503, 504) # Status codes to retry on
DEFAULT_REQUEST_TIMEOUT = 10 # Timeout for connecting and reading (seconds)
QUEUE_POLL_INTERVAL = 1.5 # Seconds between checking queue status
MAX_QUEUE_WAIT_TIME = 180 # Maximum seconds to wait for the queue to clear

# --- Utility Functions ---

def generate_random_string(length=16):
  """Generates a random alphanumeric string."""
  characters = string.ascii_letters + string.digits
  random_string = ''.join(random.choice(characters) for _ in range(length))
  return random_string

def round_to_multiple_of_8(n: int) -> int:
    """Rounds an integer up to the nearest multiple of 8, with a minimum of 8."""
    if not isinstance(n, int):
        raise TypeError("Input must be an integer.")
    # Using ceil division effectively achieves rounding up to the next multiple
    # max(1, ...) handles the case n <= 0 correctly -> 8
    return max(1, math.ceil(n / 8)) * 8

def parse_parameters(param_str: str) -> dict | None:
    """
    Parses a string containing parameters into a dictionary.

    Parameters in the string should follow the format: --{name} {value}
    Recognized parameters:
        --steps INT       (default: 20)
        --cfg FLOAT       (default: 7.0)
        --batch_size INT  (default: 1)
        --neg STR         (default: "low quality")
        --size INTxINT    (default: 512x512, becomes width/height, multiple of 8)
        --seed INT        (default: random integer)

    Args:
        param_str: The input string containing parameters.

    Returns:
        A dictionary with parsed parameter names as keys and their
        processed values. Returns None if parsing fails critically,
        but tries to use defaults for minor issues.
    """
    defaults = {
        "steps": 20, "cfg": 7.0, "batch_size": 1, "neg": "low quality",
        "width": 512, "height": 512, "seed": None,
    }
    result_dict = defaults.copy()
    pattern = r"--(\w+)\s+(.*?)(?=\s+--|\Z)"
    found_params = re.findall(pattern, param_str)
    parsed_params = {name: value.strip() for name, value in found_params}

    try:
        if "steps" in parsed_params:
            result_dict["steps"] = int(parsed_params["steps"])
        if "cfg" in parsed_params:
            result_dict["cfg"] = float(parsed_params["cfg"])
        if "batch_size" in parsed_params:
            result_dict["batch_size"] = int(parsed_params["batch_size"])
        if "neg" in parsed_params:
            result_dict["neg"] = parsed_params["neg"]
        if "seed" in parsed_params:
            result_dict["seed"] = int(parsed_params["seed"])
        else: # Explicitly handle random seed generation here if not provided
             result_dict["seed"] = random.randint(0, 2**32 - 1)

        if "size" in parsed_params:
            size_str = parsed_params["size"]
            parts = size_str.lower().split('x')
            if len(parts) == 2:
                w_str, h_str = parts
                width_raw = int(w_str.strip())
                height_raw = int(h_str.strip())
                result_dict["width"] = round_to_multiple_of_8(width_raw)
                result_dict["height"] = round_to_multiple_of_8(height_raw)
                # Optional info about rounding removed
                # if result_dict["width"] != width_raw or result_dict["height"] != height_raw:
                #      pass # Removed print statement
            else:
                raise ValueError("Size format must be INTxINT (e.g., 1024x768)")

    except (ValueError, TypeError):
        # Attempted parsing failed for one or more params.
        # Silently revert to defaults for potentially affected values.
        # This behavior might mask errors, consider raising an exception
        # if strict parsing is required.
        if "size" in parsed_params: # If size parsing failed, reset width/height
             result_dict["width"] = defaults["width"]
             result_dict["height"] = defaults["height"]
        # Reset other potentially failed conversions to default if needed
        if "steps" in parsed_params and not isinstance(result_dict["steps"], int):
            result_dict["steps"] = defaults["steps"]
        if "cfg" in parsed_params and not isinstance(result_dict["cfg"], float):
            result_dict["cfg"] = defaults["cfg"]
        if "batch_size" in parsed_params and not isinstance(result_dict["batch_size"], int):
            result_dict["batch_size"] = defaults["batch_size"]
        if "seed" in parsed_params and not isinstance(result_dict["seed"], int):
            result_dict["seed"] = random.randint(0, 2**32 - 1) # Generate random if seed was invalid

        # Depending on severity, you might want to return None here
        # return None

    return result_dict


def replace_text_in_string(template_string: str, replacements: dict) -> str:
    """
    Replaces placeholders (keys of replacements dict, converted to uppercase)
    in a template string with values from the dictionary using word boundaries.
    """
    modified_string = template_string
    for placeholder, value in replacements.items():
        replacement_value_str = str(value)
        # Match uppercase placeholder surrounded by word boundaries
        word_boundary_placeholder = rf'\b{re.escape(placeholder.upper())}\b'
        modified_string = re.sub(word_boundary_placeholder, replacement_value_str, modified_string)
    return modified_string


# --- Enhanced Request Handling ---

def create_session_with_retries(
    retries: int = DEFAULT_RETRY_COUNT,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    status_forcelist: tuple = DEFAULT_STATUS_FORCELIST,
    session: requests.Session = None,
) -> requests.Session:
    """
    Creates a requests Session configured with automatic retries.
    """
    session = session or requests.Session()
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

# --- Main Execution Logic ---

def execute_easy(
    url: str,
    positive_prompt: str,
    parameters: str,
    shiro_code: str,
    session: requests.Session = None # Allow passing a pre-configured session
) -> list[str] | None:
    """
    Executes a Shiro task on a remote server with retries and robust handling.

    Args:
        url: Base URL of the Shiro server (e.g., "http://127.0.0.1:8188").
        positive_prompt: The positive prompt string.
        parameters: String containing CLI-like parameters (e.g., "--steps 30 --cfg 8").
        shiro_code: The template Shiro code string with placeholders.
        session: An optional pre-configured requests.Session.

    Returns:
        A list of image URLs on success, None on failure after retries or errors.
    """

    # --- 1. Setup Session and Parse Parameters ---
    request_session = session or create_session_with_retries()
    params_dict = parse_parameters(parameters)
    if params_dict is None:
        # Optionally print a critical error message here if needed, but generally
        # returning None signals the failure to the caller.
        # print("Critical error: Failed to parse parameters.")
        return None

    params_dict["positive"] = positive_prompt
    filename = generate_random_string()
    params_dict["name"] = filename # Use this unique name for tracking

    # --- 2. Prepare Shiro Code ---
    try:
        shiro_code_final = replace_text_in_string(shiro_code, params_dict)
        final_payload = shiro.parse_code_str(url, shiro_code_final)
        if final_payload is None: # parse_code_str might return None on error
             # print("Error: Failed to parse Shiro code string into valid payload.") # Removed print
             return None
    except Exception:
        # print(f"Error preparing Shiro code: {e}") # Removed print
        return None

    # --- 3. Submit Prompt ---
    prompt_url = f"{url.rstrip('/')}/prompt"
    try:
        response = request_session.post(
            prompt_url,
            json={"prompt": final_payload},
            timeout=DEFAULT_REQUEST_TIMEOUT
        )
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx) after retries
    except requests.exceptions.RequestException:
        # print(f"Error submitting prompt to {prompt_url} after retries: {e}") # Removed print
        return None
    except Exception: # Catch other potential errors during submission
        # print(f"An unexpected error occurred during prompt submission: {e}") # Removed print
        return None


    # --- 4. Poll Queue Status ---
    queue_url = f"{url.rstrip('/')}/queue"
    start_time = time.time()
    try:
        while True:
            # Check for overall timeout
            if time.time() - start_time > MAX_QUEUE_WAIT_TIME:
                # print(f"Queue polling timed out after {MAX_QUEUE_WAIT_TIME} seconds for job {filename}.") # Removed print
                return None

            try:
                q_response = request_session.get(queue_url, timeout=DEFAULT_REQUEST_TIMEOUT)
                q_response.raise_for_status()
                q_data = q_response.json()

                # Basic validation
                if 'queue_running' not in q_data:
                     # print(f"Unexpected queue response format: {q_data}") # Removed print
                     return None

                running_jobs = q_data.get('queue_running', [])
                if not running_jobs:
                    break # Exit the polling loop - queue is empty

                # Minimal queue status print removed
                # print(f"Queue status: {len(running_jobs)} running.")
                time.sleep(QUEUE_POLL_INTERVAL)

            except requests.exceptions.RequestException:
                # Retries handled by session, this means retries failed
                # print(f"Error polling queue {queue_url} after retries: {e}") # Removed print
                return None
            except ValueError: # Includes JSONDecodeError
                # print(f"Error decoding queue JSON response from {queue_url}: {e}") # Removed print
                # Decide if this is fatal or worth a short wait/retry
                time.sleep(QUEUE_POLL_INTERVAL * 2) # Wait a bit longer before next poll attempt
            except Exception:
                # print(f"An unexpected error occurred during queue polling: {e}") # Removed print
                return None

    except Exception: # Catch potential errors in the outer polling loop structure itself
         # print(f"An unexpected error occurred during queue management: {e}") # Removed print
         return None


    # --- 5. Fetch Results ---
    search_url = f"{url.rstrip('/')}/shiro/image/search/{filename}"
    try:
        result_response = request_session.get(search_url, timeout=DEFAULT_REQUEST_TIMEOUT)
        result_response.raise_for_status()
        result_data = result_response.json()

        # Basic validation
        if 'file_list' not in result_data or not isinstance(result_data['file_list'], list):
             # print(f"Unexpected result format from {search_url}: {result_data}") # Removed print
             return None

        img_filenames = result_data['file_list']
        # Commented out warning about empty list - return empty list or None as appropriate
        # if not img_filenames:
            # print(f"Search for filename {filename} returned an empty file list.") # Removed print

        final_list = [f"{url.rstrip('/')}/shiro/image/{img_name}" for img_name in img_filenames]
        # Final list print removed
        # print(f"Final image URLs: {final_list}")
        return final_list

    except requests.exceptions.RequestException:
        # print(f"Error fetching results from {search_url} after retries: {e}") # Removed print
        return None
    except ValueError: # Includes JSONDecodeError
        # print(f"Error decoding result JSON response from {search_url}: {e}") # Removed print
        return None
    except Exception:
        # print(f"An unexpected error occurred fetching results: {e}") # Removed print
        return None

# --- END OF FILE base_utils.py ---