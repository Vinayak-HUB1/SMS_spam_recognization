import joblib
from Logging.setup_logger import setup_logger_



logger = setup_logger_("prediction_logs","logs/Predictionlogs.log")


def predict(data):
    try:
        model = joblib.load("models/model.pkl")
        logger.info("Prediction model is loaded")
        prediction = model.predict(data)
        logger.info(f"prediction are {prediction}")
        return prediction
    except Exception as e:
        logger.error("Error accured while prediction:" + str(e))
    