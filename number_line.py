import matplotlib.pyplot as plt


def band_to_wl(band):
    wl = 400 + (band * 0.5)
    if wl == int(wl):
        wl = int(wl)
    return wl


def band_str_to_wls(bands):
    wls = []
    for token in bands.split("|"):
        tok = int(token)
        wl = band_to_wl(tok)
        wls.append(wl)
    return wls

numbers1 = band_str_to_wls("")
numbers2 = band_str_to_wls("")

plt.figure(figsize=(10, 3))
plt.hlines(2, 400, 2500, colors='black', linewidth=2)
plt.hlines(1, 400, 2500, colors='black', linewidth=2)
plt.scatter(numbers1, [2] * len(numbers1), color='blue', zorder=3, label="Set 1")
plt.scatter(numbers2, [1] * len(numbers2), color='red', zorder=3, label="Set 2")

for num in numbers1:
    plt.text(num, 2.05, str(num), ha='center', fontsize=10, color='blue')

for num in numbers2:
    plt.text(num, 1.05, str(num), ha='center', fontsize=10, color='red')

plt.xlim(400, 2500)
plt.ylim(0.5, 2.5)
plt.axis('off')
plt.legend()
plt.show()
