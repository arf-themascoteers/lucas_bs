#bands="249|493|737|982|1226|1470|1715|1959|2203|2448|2692|2936|3181|3425|3669|3914"

bands = "386|537|681|914|1172|1322|1671|2025|2137|2225|2649|2923|3473|3687|4002"

wls = []
for token in bands.split("|"):
    tok = int(token)
    wl = 400 + (tok*0.5)
    if wl == int(wl):
        wl = int(wl)
    wls.append(wl)

print("|".join([str(x) for x in wls]))