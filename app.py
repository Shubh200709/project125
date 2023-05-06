from flask import Flask, jsonify, request
from project125 import get_predict

api = Flask(__name__)

@api.route('/',methods=['POST'])
def scrape():
    image = request.files.get('alphabets')
    predict = get_predict(image)
    jsonify({
        'prediction':predict
    })

if(__name__ == '__main__'):
    api.run(debug=True)