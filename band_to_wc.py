def band_to_wl(band):
    wl = 400 + (band * 0.5)
    if wl == int(wl):
        wl = int(wl)
    return wl


ranges = []
for i in range(7):
    start = i*600
    end = start + 599
    start = band_to_wl(start)
    end = band_to_wl(end)
    ranges.append((start, end))

print(ranges)
