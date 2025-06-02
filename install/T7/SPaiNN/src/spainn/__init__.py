import logging

logging.addLevelName(
    logging.WARNING, "\033[1;33m%s\033[1;0m" % logging.getLevelName(logging.WARNING)
)
logging.addLevelName(
    logging.CRITICAL, "\033[1;91m%s\033[1;0m" % logging.getLevelName(logging.CRITICAL)
)

from spainn import asetools
from spainn import interface
from spainn.calculator import *
from spainn.cli import *
from spainn.loss import *
from spainn.metric import *
from spainn.model import *
from spainn.multidatamodule import *
from spainn.plotting import *
from spainn.properties import SPAINN

