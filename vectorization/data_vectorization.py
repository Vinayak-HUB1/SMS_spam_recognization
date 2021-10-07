import joblib
from Logging.setup_logger import setup_logger_



logger = setup_logger_("vectorization logs","logs/vectorization.log")


def vectorization(sms):

    try:
        vectorizor = joblib.load("models/Vectorizer.pkl")
        logger.info("Vectorizer loaded")
        sms_vector = vectorizor.transform(sms)
        logger.info("sentence converted into vector")
        return sms_vector
    except Exception as e:
        logger.error("error accured while performing vectorization:" + str(e))


