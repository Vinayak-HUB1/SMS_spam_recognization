import pandas as pd
df = pd.read_csv("Data/SMSSpamCollection",sep = '\t',names=['label','message'])
print(df.head)