# logger_setup.py
import logging
import sys

class CustomFormatter(logging.Formatter):
    """Custom formatter with colored output for console display"""
    
    COLORS = {
        'DEBUG': '\033[94m',  # Blue
        'INFO': '\033[92m',   # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',   # Red
        'CRITICAL': '\033[95m',  # Magenta
        'RESET': '\033[0m'     # Reset color
    }
    
    def format(self, record):
        # Save the original format
        log_fmt = self._style._fmt
        
        # Apply the color for the log level
        if record.levelname in self.COLORS:
            self._style._fmt = f"{self.COLORS[record.levelname]}%(asctime)s - %(levelname)s - %(message)s{self.COLORS['RESET']}"
        else:
            self._style._fmt = "%(asctime)s - %(levelname)s - %(message)s"
            
        # Call the original formatter to do the grunt work
        result = super().format(record)
        
        # Restore the original format
        self._style._fmt = log_fmt
        
        return result

def setup_logger(name="pubmed_rag", log_file="PubMedRagMain.log", level=logging.DEBUG):
    """Set up a custom logger that only logs messages from your code."""
    # Create a logger with the given name
    logger = logging.getLogger(name)
    
    # Only set up the logger once
    if logger.handlers:
        return logger
        
    logger.propagate = False
    
    # Set the logging level
    logger.setLevel(level)
    
    # Create handlers
    handlers = []
    
    # Create console handler with colored formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    handlers.append(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        handlers.append(file_handler)
    
    # Add handlers to logger
    for handler in handlers:
        logger.addHandler(handler)
    
    # Disable propagation of logs from other libraries
    # Set the root logger level to a very high value
    logging.getLogger().setLevel(logging.CRITICAL + 1)
    
    return logger

# Create a default logger instance
logger = setup_logger()