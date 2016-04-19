from keras.layers import LSTM,GRU,Dense
from keras.models import Sequential
from utils import load_data
import cPickle as pickle

X_train,y_train=load_data('data/RNN_train.txt')
X_test,y_test=load_data('data/RNN_test.txt')
n_input=5
n_hidden=6
n_output=1

print X_train[1]
print y_train[1]
model=Sequential()
model.add(LSTM(output_dim=n_hidden,input_dim=5,input_length=3,return_sequences=False))
model.add(Dense(input_dim=n_hidden,output_dim=n_output,activation='sigmoid'))
model.compile(loss='mse',optimizer='rmsprop')

model.fit(X=X_train,y=y_train,nb_epoch=20,batch_size=1,show_accuracy=True)
prediction=model.predict(X_test)
with open('result.pkl','w')as f:
    pickle.dump(prediction,f)

