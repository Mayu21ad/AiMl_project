# from flask import Flask, request, render_template
# import numpy as np
# import pandas as pd

# from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# application = Flask(__name__)

# app = application


# @app.route('/')
# def index():
#     return render_template('index.htm')


# @app.route('/predictdata', methods=['GET', 'POST'])
# def predict_datapoint():
#     if request.method == 'GET':
#         return render_template('home.htm')
#     else:
#         # Collect input data from the form
#         data = CustomData(
#             feature_1=float(request.form.get('feature_1')),
#             feature_2=float(request.form.get('feature_2')),
#             feature_3=float(request.form.get('feature_3')),
#             feature_4=float(request.form.get('feature_4')),
#             feature_5=float(request.form.get('feature_5'))
#         )

#         # Convert to DataFrame
#         pred_df = data.get_data_as_data_frame()
#         print(pred_df)
#         print("Before Prediction")

#         # Select the model type (for example, 'pkl' for the traditional model, 'cnn_lstm' for CNN-LSTM)
#         model_type = request.form.get('model_type')

#         # Instantiate the prediction pipeline with the correct model type
#         predict_pipeline = PredictPipeline(model_type=model_type)

#         print("Mid Prediction")
#         results = predict_pipeline.predict(pred_df)
#         print("After Prediction")

#         return render_template('home.html', results=results[0])


# if __name__ == "__main__":
#     app.run(host="0.0.0.0")
import os

# Set the environment variable to disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


@app.route('/')
def index():
    return render_template('index.htm')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.htm')
    else:
        # Collect input data from the form
        data = CustomData(
            feature_1=float(request.form.get('feature_1')),
            feature_2=float(request.form.get('feature_2')),
            feature_3=float(request.form.get('feature_3')),
            feature_4=float(request.form.get('feature_4')),
            feature_5=float(request.form.get('feature_5'))
        )

        # Convert to DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        # Select the model type (for example, 'pkl' for the traditional model, 'cnn_lstm' for CNN-LSTM)
        model_type = request.form.get('model_type')

        # Instantiate the prediction pipeline with the correct model type
        predict_pipeline = PredictPipeline(model_type=model_type)

        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")

        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0")