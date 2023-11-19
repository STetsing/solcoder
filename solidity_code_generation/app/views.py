from app.main import app
from infer import infer

from flask import request, jsonify


@app.route('/infer', methods=['POST'])
def sg_infer():
    return jsonify(infer(request.data))
