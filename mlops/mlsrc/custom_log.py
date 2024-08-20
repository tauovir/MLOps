import logging
import datetime
import time
from pyspark.sql import SparkSession
from pyspark.dbutils import DBUtils

class DatabricksUtils:

    def __init__(self):
        self.spark = SparkSession.getActiveSession()
        self.dbutils = DBUtils (self.spark)
        
    def get_dbutils (self) -> DBUtils:
        return self.dbutils

class CustomLogger:
    """
    This is a custom log class
    """
    def __init__(self) :
        self.logger = None
        self.temp_log_file = None
        self.log_file_name = None
        self.file_handler = None

    def get_logger(self, logfile_prefix = 'logging', logger_name = 'custom_log',enable_file_log = False) :
        """
        Description: This method set the configurat ion for logging and return the logger object.
        Since, Script runs on driver node, it saves log file in tmp folder.
        """
        file_date = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d')
        log_dir = '/tmp/'
        self.enable_file_log = enable_file_log
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(name)s:%(levelname)s:%(asctime)s:%(message)s')
        #File handler logging setting.
        if self.enable_file_log:
            self.log_file_name = f'{logfile_prefix}_{file_date}.log'
            self.temp_log_file = f'{log_dir}{self.log_file_name}'
            self.file_handler = logging.FileHandler(self.temp_log_file, mode='a')
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)
        # Console logging Setting
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(stream_handler)
        return self.logger
    
    def save_logs(self, log_file_path):

        """
        Description: This method move log file from /tmp folder to mount location and shut down the logger.
        """
        if self.enable_file_log:
            dbutils = DatabricksUtils().get_dbutils()
            filename = f"{log_file_path}{self.log_file_name}" if log_file_path [-1] == '/' else f"{log_file_path}/{self.log_file_name}"
            
            self.logger.info(f"Temp file: {self.temp_log_file}")
            self.logger.info(f"Log file: {filename}")

            dbutils.fs.mv(f'file:{self.temp_log_file}', filename)
            self.logger.info( f"Log file: {self.log_file_name} saved sucessfully")
            logging.shutdown ()
            logging._removeHandlerRef(self.file_handler)
        else:
            self.logger.warning ("Enable file logging to write log into file.")

if __name__ == "__main__":
    log_obj = CustomLogger()
    logger = log_obj.get_logger(logfile_prefix = 'logging', logger_name = 'custom_log',enable_file_log = True)
    logger.info("This is a test log")
    
