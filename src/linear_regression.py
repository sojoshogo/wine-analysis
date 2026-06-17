
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\iot2026\wine-analysis\data\wine.csv')

x = df['alcohol'].to_numpy()
y = df['quality'].to_numpy()

a = (np.mean(x * y) - np.mean(x) * np.mean(y)) / (np.mean(x**2) - np.mean(x)**2)
b = np.mean(y) - a * np.mean(x)

print(f"y = {a:.4f} * alcohol + {b:.4f}")

x_line = np.linspace(x.min(), x.max(), 100)
plt.plot(x, y, '.', alpha=0.4, label='data')
plt.plot(x_line, a * x_line + b, 'r-', label=f'y = {a:.2f}x + {b:.2f}')
plt.xlabel('alcohol')
plt.ylabel('quality')
plt.title('Linear Regression')
plt.legend()
plt.tight_layout()
plt.savefig('linear_regression_result.png', dpi=150)
plt.show()
