import matplotlib.pyplot as plt

numbers1 = [float(i) for i in "524.5|646.5|768.5|891|1013|1135|1257.5|1379.5|1501.5|1624|1746|1868|1990.5|2112.5|2234.5|2357".split("|")]
numbers2 = [float(i) for i in "593|668.5|740.5|857|986|1061|1235.5|1412.5|1468.5|1512.5|1724.5|1861.5|2136.5|2243.5|2401".split("|")]

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
