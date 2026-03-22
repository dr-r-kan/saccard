import os
import warnings
import logging

# Configure backend/runtime logging before importing submodules that pull in
# TensorFlow/MediaPipe.
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('GLOG_minloglevel', '3')

logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings(
	'ignore',
	message=r'.*tf\.losses\.sparse_softmax_cross_entropy is deprecated.*',
)
warnings.filterwarnings(
	'ignore',
	message=r'.*SymbolDatabase\.GetPrototype\(\) is deprecated.*',
	category=UserWarning,
)

import pyVHR.extraction
import pyVHR.BVP
import pyVHR.BPM
import pyVHR.plot
import pyVHR.utils
import pyVHR.deepRPPG

