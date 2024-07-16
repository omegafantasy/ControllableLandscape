from dotenv import load_dotenv
import os
import numpy as np
from openai import OpenAI
import threading
from multiprocessing import Pool
from functools import partial
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import matplotlib
from matplotlib import pyplot as plt
import random
from tqdm import tqdm
import argparse
from copy import deepcopy
import json
import ast
from typing import Tuple, List, Callable
import datetime
from perlin_numpy import generate_fractal_noise_2d
from time import sleep, time
from scipy import interpolate as interp
from shapely.geometry import *
from shapely.affinity import *
from shapely.ops import *
from math import *
import cv2
import sys

load_dotenv(".env", override=True)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
matplotlib.use("Agg")

NUM_WORKERS = 16

seed = 0

W = 20
H = 15
RL = 20  # real length (meters) per unit
MAP_W = 1024
MAP_H = int((MAP_W * (H + 2) / (W + 2)) // 4 * 4)
MAX_HEIGHT = 60  # meters

WATER_HEIGHT = 0.0825
MAIN_ROAD_WIDTH = 4
SUB_ROAD_WIDTH = 2

"""
W*H units, (W+1)*(H+1) corners, 2*W*H edges
content is always located in a unit
roads are located in edges
entrances are located in corners
"""

POPULATION_SIZE = 100
PRESERVE_BEST_SIZE = 5
MUTATION_RATE = 0.7
CROSSOVER_RATE = 0.9
FIX_RATE = 0.1
MAX_GENERATION = 100

MUTATION_MINL = 2  # the minimum length of the mutated block
MUTATION_MAXL = 4  # the maximum length of the mutated block
REGION_MUTATION_RATE = 0.2  # whether to mutate a whole region or a rectanglular block
MUTATION_CONSISTENT_RATE = 0.5  # whether the mutated block should be consistent with its adjacent blocks

CROSSOVER_APPROX_RATE = 0.5  # whether to use the approximate crossover method
CROSSOVER_APPROX_ACCEPT_RATE = 0.6  # the accept rate of the approximate crossover method

color_set = [
    (
        (i % 4 + 1) * 60 / 255,
        ((i + 1) % 3 + 1) * 80 / 255,
        ((i + 2) % 5 + 1) * 50 / 255,
    )
    for i in range(200)
]
terrain_label_set = ["Unused", "Aquatic", "Terrestrial", "Artificial", "Elevated"]
content_marker_set = [",", "s", "v", "^", "*"]
content_label_set = ["None", "Basic", "Low-growing", "Tall-growing", "Architectural"]
