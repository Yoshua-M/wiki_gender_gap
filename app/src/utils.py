"""
Utility functions for project functionality
"""
import json


class Kernel:

    def __init__(self):
        with open('src/settings.json', 'r') as f:
            self.config = json.load(f)

