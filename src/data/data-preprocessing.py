import re
import os
import pandas as pd
from src.util import load_data, save_data

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)  
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText) 
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df['Resume'] = df['Resume'].apply(lambda x: cleanResume(x))
        return df
    except Exception as e:
        print(e)
        raise

if __name__ == "__main__":
    train_data = load_data(data_url=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/raw/train.csv'))
    test_data = load_data(data_url=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/raw/test.csv'))

    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)
    save_data(train_data, test_data, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data'), mode='processed')