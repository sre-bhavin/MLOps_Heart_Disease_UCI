import sys

def get_detailed_error_info(error, error_detail: sys):
    """Extracts filename and line number from the exception."""
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    # Return as a structured string that will be placed in the JSON 'message' field
    return f"Error in [{file_name}] at line [{line_number}]: {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = get_detailed_error_info(error_message, error_detail)

    def __str__(self):
        return self.error_message