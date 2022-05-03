import sys
import psutil
from multiprocessing import cpu_count
import os
def env_config():
    p = psutil.Process(os.getpid())
    try:
        p.set_cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        try:
            p.cpu_affinity(list(range(cpu_count())))
        except AttributeError:
            pass

def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()


