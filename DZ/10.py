#1) Провести дисперсионный анализ для определения того, есть ли различия среднего роста
# среди взрослых футболистов, хоккеистов и штангистов.
#Даны значения роста в трех группах случайно выбранных спортсменов:
#Футболисты: 173, 175, 180, 178, 177, 185, 183, 182.
#Хоккеисты: 177, 179, 180, 188, 177, 172, 171, 184, 180.
#Штангисты: 172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170.

import numpy as np
from scipy import stats
fp = np.array([173, 175, 180, 178, 177, 185, 183, 182], dtype = np.float64)
hp = np.array([177, 179, 180, 188, 177, 172, 171, 184, 180], dtype = np.float64)
wl = np.array([172, 173, 169, 177, 166, 180, 178, 177, 172, 166, 170], dtype = np.float64)

n1 = fp.shape[0]
n2 = hp.shape[0]
n3 = wl.shape[0]

fp_mean = fp.mean()
hp_mean = hp.mean()
wl_mean = wl.mean()

print(f"Среднее значение роста футболистов = {fp_mean:.2f}")
print(f"Среднее значение роста хоккеистов = {hp_mean:.2f}")
print(f"Среднее значение роста штангистов = {wl_mean:.2f}")

y = np.concatenate([fp, hp, wl])
y_mean = y.mean()
print(f"Среднее значение роста спортсменов = {y_mean:.2f}")

S2_F = n1 * (fp_mean - y_mean) ** 2 + n2 * (hp_mean - y_mean) ** 2 + n3 * (wl_mean - y_mean) ** 2
S2_res = ((fp - fp_mean) ** 2).sum() + ((hp - hp_mean) ** 2).sum() + ((wl - wl_mean) ** 2).sum()
S2 = ((y - y_mean) ** 2).sum()

print(f" Sf^2 = {S2_F:.2f}")
print(f" Sres^2 = {S2_res:.2f}")
print(f" S^2 = Sf^2 + Sres^2")
print(f"{S2:.2f} = {S2_F + S2_res:.2f}")

k = 3
n = n1 + n2 + n3

k1 = k - 1
k2 = n - k

sigma2_F = S2_F / k1
sigma2_res = S2_res / k2

print(f"Факторная дисперсия = {sigma2_F:.2f}")
print(f"Остаточная дисперсия = {sigma2_res:.2f}")

T = sigma2_F / sigma2_res
print(f"Значение статистики = {T:.2f}")

f = stats.f_oneway(fp, hp, wl)
print(f)
alpha = 0.05

F_crit = stats.f.ppf(1 - alpha, k1, k2)
print(f"Критическое значение = {F_crit:.2f}")

#Так как T > F_crit делаем вывод, что отличие среднего роста спортсменов является статистически значимым
