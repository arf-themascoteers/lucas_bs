import matplotlib.pyplot as plt
import os


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

# numbers1 = band_str_to_wls("466|927|1389|1850|2312|2773|3235|3696")
# numbers2 = band_str_to_wls("674|897|1411|2021|2308|2787|3467|3606")

numbers1 = band_str_to_wls("249|493|737|982|1226|1470|1715|1959|2203|2448|2692|2936|3181|3425|3669|3914")
numbers2 = band_str_to_wls("392|538|684|912|1138|1310|1674|2025|2136|2218|2630|2923|3473|3475|3664|3990")

print(numbers1)
print(numbers2)

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
os.makedirs("wls", exist_ok=True)
plt.savefig("wls/16.png")
plt.show()
