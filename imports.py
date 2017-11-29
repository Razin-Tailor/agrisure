from flask import Flask, render_template, request, jsonify

import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
# import matplotlib.pyplot as plt
