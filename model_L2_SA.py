#Regression L2 with analytical solution
#importar librerías
from data_load_L1yL2 import X_train, X_test, y_train, y_test
import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error, r2_score


w = np.zeros(X_train.shape[1], dtype=np.float64)
b = 0.3
a = 0.1
N = len(y_train)
lmb = 0.00001
I = np.eye(X_train.shape[1])
ones = np.ones(N)


#metricas del modelo
def evaluate_model(y_true, y_pred):
  MSE = mean_squared_error(y_true, y_pred)
  RMSE = root_mean_squared_error(y_true, y_pred)
  MAE = mean_absolute_error(y_true, y_pred)
  R2 = r2_score(y_true, y_pred)
  return MSE, RMSE, MAE, R2

#promedio
X_mean_train = np.mean(X_train, axis=0)
y_mean_train = np.mean(y_train)

#centrado
Xc_train = X_train - X_mean_train
yc_train = y_train - y_mean_train

#solucion analítica
SA_w = np.linalg.inv((Xc_train.T @ Xc_train) + (N * lmb * I)) @ (Xc_train.T @ yc_train)
SA_b = y_mean_train - (X_mean_train.T @ SA_w)

#funcion de perdida
y_hat = X_train @ SA_w + SA_b  #yc_hat = (353,)
error = y_hat - y_train #error = (353,)
L = (1 / N) * (error.T @ error) + (lmb * (SA_w.T @ SA_w))

#prediccion
pred = X_test @ SA_w + SA_b

#resultados
print(f"funcion de perdida  {L:.4f} \n")
MSE, RMSE, MAE, R2 = evaluate_model(y_test, pred)
print(f"MSE:  {MSE:.4f}")
print(f"RMSE: {RMSE:.4f}")
print(f"MAE:  {MAE:.4f}")
print(f"R²:   {R2:.4f}")


#MSE:  2889.3534
#RMSE: 53.7527
#MAE:  42.8180
#R²:   0.4546