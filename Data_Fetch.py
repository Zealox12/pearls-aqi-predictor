import requests
import json
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
API_KEY = os.getenv('API_KEY')
