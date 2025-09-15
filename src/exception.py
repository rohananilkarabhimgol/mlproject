import sys

def error_message_details(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()  # Get exception traceback info
    file_name = exc_tb.tb_frame.f_code.co_filename  # Extract filename
    error_message = "Error occured in python script name [{0}] line number [{1}] error message[{2}]".format(file_name,exc_tb.tb_lineno,str(error))
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        # 1. Initialize parent Exception with basic message
        super().__init__(error_message)  
        """
        # When you raise this exception:
        raise CustomException("File not found", sys.exc_info())
                                ↑
                        This becomes error_message parameter
        """
        
        # 2. Create detailed message using external function
        self.error_message = error_message_details(error_message, error_detail)
        """
        This calls the function and stores the result
        self.error_message = error_message_details(error_message, error_detail)
                           ↑
                   Stores the RETURNED string
        """
    
    def __str__(self):
        # 3. When exception is printed, show the detailed message
        return self.error_message

"""
# When this executes:
self.error_message = error_message_details(error_message, error_detail)

# What happens:
# 1. The constructor's 'error_message' parameter value is passed
# 2. It becomes the 'error' parameter in error_message_details()
# 3. Inside the function, a NEW local variable 'error_message' is created
# 4. This local variable is returned and stored in self.error_message
"""

import logging
import logger # This sets up logging
if __name__ =="__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.info("Zero Division Error!")
        logging.warning("This is a warning you cannot divide a number by zero!")
        
        # Create the custom exception
        custom_exception = CustomException(e, sys)
        
        # Log it to file
        logging.error(str(custom_exception))  # ← Save to log file
        
        # Also raise it for console
        raise custom_exception  # ← Show in console