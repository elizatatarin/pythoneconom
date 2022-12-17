from colorama import Fore
import numpy as np
from sklearn.linear_model import LinearRegression

print(f"18-й варіант")

print(f"Подамо x як двовимірний масив:")
x = np.array([11.5, 12.6, 14.6, 15.2, 17.2, 19.6, 20.3, 22.7, 24.5, 26.7]).reshape((-1, 1))
y = np.array([7.9, 10.6, 11.8, 12.3, 15.5, 17.8, 18.9, 21.7, 22.3, 23.5])

model = LinearRegression()
print(model.fit(x, y))

r_sq = round(model.score(x, y), 4)
print(f"Коефіцієнт детермінації (𝑅²): {r_sq}")

print(f"intercept (a, 𝑏₀): {round(model.intercept_, 4)}")
print(f"slope (b, 𝑏₁): {model.coef_}")

print(f"Подамо і y як двовимірний масив:")

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print(f"intercept (a, 𝑏₀): {new_model.intercept_}")
print(f"slope (b, 𝑏₁): {new_model.coef_}")

y_pred = model.predict(x)
print(f"predicted response (g(xi)): \n{y_pred}")

y_pred = model.intercept_ + model.coef_ * x
print(f"predicted response (g(xi)):\n{y_pred}")

x_new = np.arange(5).reshape((-1, 1))
print(x_new)
y_new = model.predict(x_new)
print(y_new)

print(Fore.LIGHTCYAN_EX + f"Висновки:"
                 f"\n (𝑅²): {r_sq},"
                 f"\n (a, 𝑏₀): {round(model.intercept_, 4)},"
                 f"\n (b, 𝑏₁): {model.coef_}")


