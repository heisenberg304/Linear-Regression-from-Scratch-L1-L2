#L2 Ridge regression
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

#forward pass
#calcular error
def grad_error(w,b):
  error = (X_train @ w  + b) - y_train #error = (353,)
  L = (1 / N) * (error.T @ error) + lmb * (w.T @ w)
  lmb_grad = 2 * lmb * w 
  L_w = ((2 / N) * (X_train.T @ error)) + lmb_grad
  L_b = (2 / N) *  error.sum()
  return L_w, L_b, L



#calcular derivadas
for epoch in range(3001):
  W = w
  B = b
  grad_w, grad_b, L= grad_error(w,b)

#actualizar pesos
  w = w - (a * grad_w)
  b = b - (a * grad_b)

#repetir
  if epoch % 1000 == 0:
    print(f"iter {epoch}, {W} --> {w} ----- {B} --> {b} ")
    print(f"funcion de perdida  {L:.4f} \n")

#testear
  if epoch % 3000 == 0:
    pred = (X_test @ w) + b
    MSE, RMSE, MAE, R2 = evaluate_model(y_test, pred)
    print(f"MSE:  {MSE:.4f}")
    print(f"RMSE: {RMSE:.4f}")
    print(f"MAE:  {MAE:.4f}")
    print(f"R²:   {R2:.4f}")


#MSE:  2870.4058
#RMSE: 53.5762
#MAE:  43.1716
#R²:   0.4582