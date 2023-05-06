import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from PIL import Image

x = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
# print(pd.Series(y).value_counts())
classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
nclasses = len(classes)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=2500,train_size=7500,random_state=0)
x_train_scale = x_train/255.0
x_test_scale = x_test/255.0

logreg = LogisticRegression(solver='saga',multi_class = 'multinomial')
logreg.fit(x_train_scale,y_train)

def get_predict(image):
    img = Image.open(image)
    gray_img = img.convert('L')
    re_img = gray_img.resize((28,28), Image.ANTIALIAS)

    fil_img = 20
    min_pix = np.percentile(re_img, fil_img)

    clip_img = np.clip((re_img-min_pix), 0, 255)
    max = np.max(re_img)
    arr_img = np.asarray(clip_img)/max

    test = np.array(clip_img).reshape(1,784)
    predict = logreg.predict(test)

    return predict[0]