#importar librerías
from data_load_L1yL2 import X_train, X_test, y_train, y_test
import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score
 
 
#X_train.shape (353, 10), y_train.shape (353,)
#X_test.shape (89, 10), y_test.shape (89,)
w = np.zeros(X_train.shape[1], dtype=np.float64)
b = 0.3
a = 0.1
lmb = 0.1
N = len(y_train)
#metricas del modelo
def evaluate_model(y_true, y_pred):
  MSE = mean_squared_error(y_true, y_pred)
  RMSE = root_mean_squared_error(y_true, y_pred)
  MAE = mean_absolute_error(y_true, y_pred)
  R2 = r2_score(y_true, y_pred)
  return MSE, RMSE, MAE, R2


error = (X_train @ w  + b) - y_train #error = (353,)
g = (1 / N) * (error.T @ error)
L =  g + lmb * (w.T @ w)
lmb_grad = 2 * lmb * w 
L_w = ((2 / N) * (X_train.T @ error)) + lmb_grad
L_b = (2 / N) *  error.sum()

print("error shape:", error.shape)
print("L shape:", L)
print("L_w shape:", L_w.shape)
print("L_b shape:", L_b.shape)

w = w - (a * L_w)
b = b - (a * L_b)
print("w shape:", w.shape)
print("b shape:", b.shape)

def soft_thresholding(w, lmb, b, a):
    g, error = provisional_grad(w,b)
    #ISTA:
    w_1 = w - a * g
    w_new = []
    for i in range(len(w_1)):
        if w_1[i] > (a * lmb):
            w_new.append(w_1[i] - a*lmb)
        elif w_1[i] < -(a * lmb):
            w_new.append(w_1[i] + a*lmb)
        elif np.abs(w_1[i]) <= (a * lmb):
            w_new.append(0)
    return np.array(w_new)

#forward pass
def provisional_grad(w,b):
    error = (X_train @ w  + b) - y_train #error = (353,)
    g = (1 / N) * (error.T @ error)
    return g, error

w_new = soft_thresholding(w, lmb, b, a)
print("w_new shape:", w_new.shape)

error = (X_train @ w_new  + b) - y_train