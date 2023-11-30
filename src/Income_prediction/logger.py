import os
from datetime import datetime
import logging
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_path=os.path.join(os.getcwd(),"logs")

os.makedirs(log_path,exist_ok=True)

LOGFILEPATH=os.path.join(log_path,LOG_FILE)

logging.basicConfig(level=logging.INFO,
                    filename=LOGFILEPATH,
                    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s")

if __name__=="__main__":
    logging.info("Here again i am testing")