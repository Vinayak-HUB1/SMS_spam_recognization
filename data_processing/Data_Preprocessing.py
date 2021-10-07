from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from Logging.setup_logger import setup_logger_
import re


logger = setup_logger_("processing logs","logs/Processing.log")
ps = PorterStemmer()
corpus = []


class Processing:
    def __init__(self):
       pass


    def Process(self,sms):
        try:
            for i in range(0, len(sms)):
                sms_ = re.sub('[^a-zA-Z]', ' ',sms[i])
                logger.info("unwanted words removed")
                sms_ = sms_.lower()
                sms_ = sms_.split()
                logger.info("data splitting and lowering completed")
                sms_ = [ps.stem(word) for word in sms_ if not word in stopwords.words('english') ]
                logger.info("data stemming completed")
                sms_ = ' '.join(sms_)
                corpus.append(sms_)
                return corpus
        except Exception as e:
            logger.error("Exception accured while processing:" + str(e))

