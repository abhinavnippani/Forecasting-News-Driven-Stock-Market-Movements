

def XGBoost_Algo(X_train,Y_train,X_test):
    
    ##### Import libraries ######
    
    from sklearn.multiclass import OneVsRestClassifier
    from xgboost import XGBClassifier
    import warnings
    warnings.filterwarnings("ignore")
    
    ###### Model ########
    
    # Define the Model
    regressor = OneVsRestClassifier(XGBClassifier())
    # Train the model
    regressor.fit(X_train, Y_train)
    # Predict the values
    Y_pred = regressor.predict(X_test)
    
    ##########
    
    return Y_pred

def DecisionTree_Algo(X_train,Y_train,X_test):
    
    ##### Import libraries ######
    
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.tree import DecisionTreeClassifier
    import warnings
    warnings.filterwarnings("ignore")
    
    ###### Model ########
    
    # Define the Model
    regressor = OneVsRestClassifier(DecisionTreeClassifier())
    # Train the model
    regressor.fit(X_train, Y_train)
    # Predict the values
    Y_pred = regressor.predict(X_test)
    
    ##########
    
    return Y_pred

def LinearSVM_Algo(X_train,Y_train,X_test):
    
    ##### Import libraries ######
    
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    import warnings
    warnings.filterwarnings("ignore")
    
    ###### Model ########
    
    # Define the Model
    regressor = OneVsRestClassifier(SVC(kernel = 'linear'))
    # Train the model
    regressor.fit(X_train, Y_train)
    # Predict the values
    Y_pred = regressor.predict(X_test)
    
    ##########
    
    return Y_pred

def KernelSVM_Algo(X_train,Y_train,X_test):
    
    ##### Import libraries ######
    
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC
    import warnings
    warnings.filterwarnings("ignore")
    
    ###### Model ########
    
    # Define the Model
    regressor = OneVsRestClassifier(SVC())
    # Train the model
    regressor.fit(X_train, Y_train)
    # Predict the values
    Y_pred = regressor.predict(X_test)
    
    ##########
    
    return Y_pred

def KNN_Algo(X_train,Y_train,X_test):
    
    ##### Import libraries ######
    
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    import warnings
    warnings.filterwarnings("ignore")
    
    ###### Model ########
    
    # Define the Model
    regressor = OneVsRestClassifier(KNeighborsClassifier())
    # Train the model
    regressor.fit(X_train, Y_train)
    # Predict the values
    Y_pred = regressor.predict(X_test)
    
    ##########
    
    return Y_pred

def RandomForest_Algo(X_train,Y_train,X_test):
    
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.ensemble import RandomForestRegressor
    import warnings
    warnings.filterwarnings("ignore")
    
    regressor = OneVsRestClassifier(RandomForestRegressor())
    regressor.fit(X_train, Y_train)
    Y_pred = regressor.predict(X_test)
    
    return Y_pred


def BiLSTM_Algo(X_train,Y_train,X_test):
    
    import numpy as np    
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(64)))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, Y_train, epochs=10,batch_size = 20, verbose=2, shuffle=False)
    
    Y_pred = model.predict(X_test)
    
    #print(Y_pred)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    #encoded_Y = np.eye(2)[encoded_Y]
    #print(encoded_Y)
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    #print(encoded_Y)
    
    #Y_pred = np_utils.to_categorical(np.argmax(Y_pred,axis=1))
    
    
    return encoded_Y

def BiLSTM_2_Algo(X_train,Y_train,X_test):
    
    import numpy as np    
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow as tf
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(64,return_sequences = True)))
    model.add(layers.Bidirectional(layers.LSTM(64)))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, Y_train, epochs=10,batch_size = 20, verbose=2, shuffle=False)
    
    Y_pred = model.predict(X_test)
    
    #print(Y_pred)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    #encoded_Y = np.eye(2)[encoded_Y]
    #print(encoded_Y)
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    #print(encoded_Y)
    
    #Y_pred = np_utils.to_categorical(np.argmax(Y_pred,axis=1))
    
    
    return encoded_Y


def LSTM_Algo(X_train,Y_train,X_test):
    
    import numpy as np  
    import keras
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow as tf
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.LSTM(64))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    model.fit(X_train, Y_train, epochs=10,batch_size = 20, shuffle=True)
    
    #print(model.summary())
    
    Y_pred = model.predict(X_test)
    
    #print(Y_pred)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    #encoded_Y = np.eye(2)[encoded_Y]
    #print(encoded_Y)
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    #print(encoded_Y)
    
    #Y_pred = np_utils.to_categorical(np.argmax(Y_pred,axis=1))
    
    
    return encoded_Y

def LSTM_2_Algo(X_train,Y_train,X_test):
    
    import numpy as np    
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.LSTM(64,return_sequences = True))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, Y_train, epochs=10,batch_size = 20, verbose=2, shuffle=False)
    
    Y_pred = model.predict(X_test)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    
    
    return encoded_Y

def BiRNN_Algo(X_train,Y_train,X_test):
    
    import numpy as np    
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.Bidirectional(layers.SimpleRNN(64)))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, Y_train, epochs=10,batch_size = 20, verbose=2, shuffle=False)
    
    Y_pred = model.predict(X_test)
    
    #print(Y_pred)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    #encoded_Y = np.eye(2)[encoded_Y]
    #print(encoded_Y)
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    #print(encoded_Y)
    
    #Y_pred = np_utils.to_categorical(np.argmax(Y_pred,axis=1))
    
    
    return encoded_Y

def BiRNN_2_Algo(X_train,Y_train,X_test):
    
    import numpy as np    
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.Bidirectional(layers.SimpleRNN(64,return_sequences = True)))
    model.add(layers.Bidirectional(layers.SimpleRNN(64)))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, Y_train, epochs=10,batch_size = 20, verbose=2, shuffle=False)
    
    Y_pred = model.predict(X_test)
    
    #print(Y_pred)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    #encoded_Y = np.eye(2)[encoded_Y]
    #print(encoded_Y)
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    #print(encoded_Y)
    
    #Y_pred = np_utils.to_categorical(np.argmax(Y_pred,axis=1))
    
    
    return encoded_Y

def RNN_Algo(X_train,Y_train,X_test):
    
    import numpy as np    
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.SimpleRNN(64))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, Y_train, epochs=10,batch_size = 20, verbose=2, shuffle=False)
    
    Y_pred = model.predict(X_test)
    
    #print(Y_pred)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    #encoded_Y = np.eye(2)[encoded_Y]
    #print(encoded_Y)
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    #print(encoded_Y)
    
    #Y_pred = np_utils.to_categorical(np.argmax(Y_pred,axis=1))
    
    
    return encoded_Y

def RNN_2_Algo(X_train,Y_train,X_test):
    
    import numpy as np    
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.SimpleRNN(64,return_sequences = True))
    model.add(layers.SimpleRNN(64))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, Y_train, epochs=10,batch_size = 20, verbose=2, shuffle=False)
    
    Y_pred = model.predict(X_test)
    
    #print(Y_pred)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    #encoded_Y = np.eye(2)[encoded_Y]
    #print(encoded_Y)
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    #print(encoded_Y)
    
    #Y_pred = np_utils.to_categorical(np.argmax(Y_pred,axis=1))
    
    
    return encoded_Y


def BiGRU_Algo(X_train,Y_train,X_test):
    
    import numpy as np    
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.Bidirectional(layers.GRU(64)))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, Y_train, epochs=10,batch_size = 20, verbose=2, shuffle=False)
    
    Y_pred = model.predict(X_test)
    
    #print(Y_pred)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    #encoded_Y = np.eye(2)[encoded_Y]
    #print(encoded_Y)
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    #print(encoded_Y)
    
    #Y_pred = np_utils.to_categorical(np.argmax(Y_pred,axis=1))
    
    
    return encoded_Y

def BiGRU_2_Algo(X_train,Y_train,X_test):
    
    import numpy as np    
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.Bidirectional(layers.GRU(64,return_sequences = True)))
    model.add(layers.Bidirectional(layers.GRU(64)))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, Y_train, epochs=10,batch_size = 20, verbose=2, shuffle=False)
    
    Y_pred = model.predict(X_test)
    
    #print(Y_pred)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    #encoded_Y = np.eye(2)[encoded_Y]
    #print(encoded_Y)
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    #print(encoded_Y)
    
    #Y_pred = np_utils.to_categorical(np.argmax(Y_pred,axis=1))
    
    
    return encoded_Y


def GRU_Algo(X_train,Y_train,X_test):
    
    import numpy as np    
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.GRU(64))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, Y_train, epochs=10,batch_size = 20, verbose=2, shuffle=False)
    
    Y_pred = model.predict(X_test)
    
    #print(Y_pred)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    #encoded_Y = np.eye(2)[encoded_Y]
    #print(encoded_Y)
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    #print(encoded_Y)
    
    #Y_pred = np_utils.to_categorical(np.argmax(Y_pred,axis=1))
    
    
    return encoded_Y

def GRU_2_Algo(X_train,Y_train,X_test):
    
    import numpy as np    
    from keras.utils import np_utils
    from sklearn.preprocessing import LabelEncoder
    from keras.models import Sequential
    from keras.layers import LSTM,Dense
    from keras import layers
    import tensorflow
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    keras.backend.clear_session()
    
    # design network
    model = Sequential()
    model.add(layers.GRU(64,return_sequences = True))
    model.add(layers.GRU(64))
    model.add(layers.Dense(Y_train.shape[1], activation='softmax'))
    #model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    
    #compile model using accuracy to measure model performance
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # fit network
    history = model.fit(X_train, Y_train, epochs=10,batch_size = 20, verbose=2, shuffle=False)
    
    Y_pred = model.predict(X_test)
    
    encoded_Y = np.argmax(Y_pred,axis=1)
    
    encoded_Y = np.array([np.eye(2)[dummy] for dummy in encoded_Y])
    encoded_Y = np.array(encoded_Y).reshape(-1,encoded_Y.shape[1])
    
    
    return encoded_Y