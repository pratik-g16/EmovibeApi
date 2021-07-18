from flask import Flask, jsonify
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
from flask_cors import CORS, cross_origin
from flask import Flask, jsonify, request
from pymongo import MongoClient
warnings.filterwarnings('ignore')
# from tensorflow.keras.preprocessing import text


app = Flask(__name__)
# CORS(app)


cors = CORS(app, resources={r"/api/*": {"origins": "*"}})


@app.route('/')
def mget():
    return "hi"


class CustomModelPrediction(object):

    def __init__(self, model, processor):
        self._model = model
        self._processor = processor

    def predict(self, instances, **kwargs):
        preprocessed_data = self._processor.transform_text(instances)
        predictions = self._model.predict(preprocessed_data)
        return predictions.tolist()

    @classmethod
    def from_path(cls, model_dir):
        import tensorflow.keras as keras
        # model = keras.models.load_model(
        #   'C:/Users/HP/Desktop/ProjectSem6/VSC/sentiment-analysis/TextSentimentAnalysis/sentiment_model.h5')
        # with open('C:/Users/HP/Desktop/ProjectSem6/VSC/sentiment-analysis/TextSentimentAnalysis/processor_state.pkl', 'rb') as f:
        # print(os.getcwd())
        model = keras.models.load_model(
            'sentiment_model.h5')
        with open('processor_state.pkl', 'rb') as f:
            processor = pickle.load(f)
        return cls(model, processor)
        # model = keras.models.load_model(
        #   os.path.join('sentiment_model.h5'))
        # with open(os.path.join('processor_state.pkl'), 'rb') as f:
        #   processor = pickle.load(f)
        # return cls(model, processor)


class TextPreprocessor(object):
    def __init__(self, vocab_size):
        self._vocab_size = vocab_size
        self._tokenizer = None

    def create_tokenizer(self, text_list):
        tokenizer = text_list.Tokenizer(num_words=self._vocab_size)
        tokenizer.fit_on_texts(text_list)
        self._tokenizer = tokenizer

    def transform_text(self, text_list):
        text_matrix = self._tokenizer.texts_to_matrix(text_list)
        return text_matrix


tag_encoder = MultiLabelBinarizer()

tag_encoder.classes_ = ['anger', 'fear', 'happiness', 'happy', 'love', 'neutral', 'relief', 'sadness',
                        'surprise', 'worry']


def predict_emotion(text_requests):

    classifier = CustomModelPrediction.from_path('.')
    # print("----------------------------------------------")
    # print(classifier)
    results = classifier.predict(text_requests)
    emotion = []
    for i in range(len(results)):
        for idx, val in enumerate(results[i]):
            if val > 0.1:
                emotion.append([val, tag_encoder.classes_[idx]])
    emotion.sort(reverse=True)
    return emotion[:2]


@app.route('/text', methods=['POST'])
@cross_origin()
def index():
    some_json = request.get_json()
    text = some_json
    print("**************************text*****************************")
    print(text['text'])
    text_requests = [text['text']]
    # return jsonify({'you sent':some_json}),201
    emotions = predict_emotion(text_requests)
    emotion1 = emotions[0]
    emotion1 = emotion1[1]
    if(emotion1 == 'happiness'):
        emotion1 = "HAPPY"
    # emotion2=emotions[1]
    # emotion2=emotion2[1]
    print(emotion1)
    client = MongoClient(
        "mongodb+srv://test:test@cluster0.nvu4g.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
    db = client.get_database('song_db')
    # print(db)
    records = db.song_records
    # print(records.count_documents({}))

    songlist = list(records.find({'emotion': emotion1.upper()}))
    # print(songlist)
    # print(songlist)
    import random as r
    finalSongs = []
    song_index = set()
    while(len(finalSongs) != 3):
        temp = {}
        temp['name'] = songlist[r.randrange(0, len(songlist))]['name']
        temp['emotion'] = songlist[r.randrange(0, len(songlist))]['emotion']
        temp['link'] = songlist[r.randrange(0, len(songlist))]['link']
        # finalSongs.append(songlist[r.randrange(0,len(songlist))])
        if(temp not in finalSongs):
            finalSongs.append(temp)
    # print("-------------")
    # print(finalSongs)
    return jsonify({'songs': finalSongs})


@app.route('/songs')
def emotion():
    # return jsonify({'about':'hi'})
    # return jsonify({'emotion':emotions})
    return jsonify({'songs': finalSongs})
    # return finalSongs


@app.route('/text', methods=['POST'])
def getText():
    if request.method == 'POST':
        text = request.form


if __name__ == '__main__':
    app.run(debug=True)
