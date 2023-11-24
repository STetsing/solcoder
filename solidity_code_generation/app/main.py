from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # allow all origins to access the api since wildcards don't work
# , [
#   "http://localhost:8080",
#   "https://remix.ethereum.org",
#   "https://remix-alpha.ethereum.org",
#   "https://remix-beta.ethereum.org",
#   "https://deploy-preview-*--remixproject.netlify.app/"
# ], allow_headers="*", allow_methods="*", supports_credentials=True)
from app.views import *
