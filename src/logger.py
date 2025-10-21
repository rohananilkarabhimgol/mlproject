"""
we use logger to capture valuable information about code execution and software or application behavior during runtime
"""
import logging
import os
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
# strftime() function lets us convert a datetime object into a formatted string using special format codes
# Creates filename like: "09_15_2023_14_30_45.log"
# Format: Month_Day_Year_Hour_Minute_Second.log

logs_path = os.path.join(os.getcwd(),"logs")
# Creates path like: "/current/working/directory/logs"

os.makedirs(logs_path,exist_ok=True)
# Creates the directory if it doesn't exist
# exist_ok=True prevents errors if directory already exists

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)
# Final path: "/current/working/directory/logs/09_15_2023_14_30_45.log"

logging.basicConfig(
    filename=LOG_FILE_PATH,# Where to save logs
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,# Log level: INFO and above
)

# if __name__ =="__main__":
#     logging.info("Logging has started")