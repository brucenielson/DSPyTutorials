def get_secret(secret_file: str) -> str:
    """
    Read a secret from a file.

    Args:
        secret_file (str): Path to the file containing the secret.

    Returns:
        str: The content of the secret file, or an empty string if an error occurs.
    """
    try:
        with open(secret_file, 'r') as file:
            secret_text: str = file.read().strip()
    except FileNotFoundError:
        print(f"The file '{secret_file}' does not exist.")
        secret_text = ""
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

    return secret_text
