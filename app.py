from flask import Flask,request,jsonify
from data_processing.Data_Preprocessing import Processing
from vectorization.data_vectorization import vectorization
from Logging.setup_logger import setup_logger_
from final_pred.Prediction import predict

app = Flask(__name__)
logger = setup_logger_("Logs","logs/Flask_app.txt")

@app.route("/predict",methods = ["POST"])
def prediction():
    if request.method == "POST":
        sms = request.json["sms"]
        data_process = Processing()
        logger.info("Preprocessing object initialized")
        sms_process = data_process.Process(sms)
        logger.info("data Preprocessing completed")
        vec = vectorization(sms_process).toarray()
        logger.info("data vectorization completed")
        prediction_ = predict(vec)
        logger.info(f"predictions are {prediction_}")
        if int(prediction_)==0:
            result = "message is not Spam"
        else:
            result  =  "message is spam"
        return jsonify({result:int(prediction_)})




    
if __name__=="__main__":
    app.run(debug=True)