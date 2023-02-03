import pickle
import sys
import numpy as np

if __name__ == '__main__':
    # take argument from cmd
    t11 = float(sys.argv[1])
    t12 = float(sys.argv[2])
    t21 = float(sys.argv[3])
    t22 = float(sys.argv[4])

    v11 = float(sys.argv[5])
    v12 = float(sys.argv[6])
    v21 = float(sys.argv[7])
    v22 = float(sys.argv[8])

    # load model and scaler
    model = pickle.load(open('model_q1.pkl', 'rb'))
    scaler_temp = pickle.load(open('scaler_temp.pkl', 'rb'))
    scaler_vib = pickle.load(open('scaler_vib.pkl', 'rb'))

    # preprocessing
    feature = np.array([[t11, t12, t21, t22, v11, v12, v21, v22]])
    feature[:, 0:4] = scaler_temp.transform(feature[:, 0:4])
    feature[:, 0:4] = scaler_vib.transform(feature[:, 4:])

    # predict data
    y = model.predict(feature)
    
    if y[0] == 0:
        print('Prediction: normal')
    elif y[0] == 1:
        print('Prediction: fault')

    