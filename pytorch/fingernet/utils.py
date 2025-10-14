import logging
import time
import sys
from pythonjsonlogger import jsonlogger

class Timer:
    """Context manager to time a code block."""
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = (self.end - self.start) * 1000  # em milissegundos

def setup_logging(rank: int = -1):
    """Configura o logger para cuspir JSON com contexto de rank."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Previne duplicação de handlers em DDP
    if logger.hasHandlers():
        logger.handlers.clear()
        
    handler = logging.StreamHandler(sys.stdout)
    
    # Adiciona o rank (GPU ID) a todos os logs. Se for -1, é a CPU ou single GPU.
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(message)s',
        rename_fields={'levelname': 'level'}
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Adiciona contexto padrão
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.rank = rank
        return record
    logging.setLogRecordFactory(record_factory)

    return logger