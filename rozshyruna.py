from colorama import Fore
import numpy as np
import statsmodels.api as sm

print(f"18-й варіант")

y = [9.08, 10.92, 12.42, 10.9, 11.52, 14.88, 15.2, 14.08, 14.48,
     14.7, 18.34, 17.22, 19.42]

x = [
    [9, 10.11, 2.29], [8.03, 12.72, 11.51], [9.66, 11.78, 11.46],
    [11.34, 14.87, 11.55], [10.99, 15.32, 14], [13.23, 16.63, 11.77],
    [14.02, 16.39, 13.74], [12.78, 17.93, 13.4], [14.14, 19.6, 14.01],
    [14.67, 18.64, 16.25], [15.35, 18.92, 16.72], [15.69, 21.22, 14.4], [17.5, 21.84, 18.19]]

x, y = np.array(x), np.array(y)

x = sm.add_constant(x)
print(f"X = \n{x}")
print(f"Y = \n{y}")

model = sm.OLS(y, x)
results = model.fit()

print(Fore.LIGHTCYAN_EX, f"Regression (Регресія) як в Exel:"
                   f"\n{results.summary()}")