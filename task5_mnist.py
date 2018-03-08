import math
import tensorflow as tf
import random

# note that this only works for a single layer of depth
INPUT_NODES = 256
HIDDEN_NODES = 30
OUTPUT_NODES = 10

# 15000 iterations is a good point for playing with learning rate
MAX_ITERATIONS = 130000


# setting this too low makes everything change very slowly, but too high
# makes it jump at each and every example and oscillate. I found .5 to be good
LEARNING_RATE = .2

