import os
import logging
from inspect import getframeinfo, currentframe
from datetime import datetime

LOGGING_FILE = "lab.log"

FORMAT = "{} @ [{}|{}] -> {}"

logging.basicConfig(filename=LOGGING_FILE, level=logging.INFO)

LOGGER = logging.getLogger(' lab ')

PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(PATH)


# print(filename)