#bands="249|493|737|982|1226|1470|1715|1959|2203|2448|2692|2936|3181|3425|3669|3914"

bands = "359|416|503|553|619|697|750|987|992|1185|1244|1455|1622|1715|1924|2139|2198|2443|2445|2653|2661|2662|2966|2960|3450|3471|3571|3571|3711|3733|3933|4035"

wls = []
for token in bands.split("|"):
    tok = int(token)
    wl = 400 + (tok*0.5)
    if wl == int(wl):
        wl = int(wl)
    wls.append(wl)

#print("|".join([str(x) for x in wls]))
print([x for x in wls])