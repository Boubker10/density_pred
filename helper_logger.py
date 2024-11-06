import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Configure un logger pour le suivi."""
    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

logger = setup_logger('app_logger', 'logs/app.log')
logger.info("Logger configuré avec succès.")
