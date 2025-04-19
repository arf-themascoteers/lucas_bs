import matplotlib.pyplot as plt
import numpy as np

targets = [8, 16, 32, 64, 128, 256, 512]
x = np.arange(len(targets))
width = 0.35

ad_fod = [0.025567008523671567, 0.019557029730651137, 0.012735128883764095, 0.009069849940749383, 0.0061983064809726885, 0.0033541600931887203, 0.0017285210658695312]
bsdr_fod = [0.04494568606716215, 0.025030217751924072, 0.0180059946563843, 0.012460053023123448, 0.01029585420948087, 0.01056201680812799, 0.008025800773500787]

ad_sod = [0.000581878706863133, 0.0010505676454760065, 0.0019683843627024394, 0.0023500523278014245, 0.002305073317380386, 0.0015673754996414037, 0.0008005993752789955]
bsdr_sod = [0.0055583479802947, 0.0032004353197932936, 0.00514629455448515, 0.007428957965655262, 0.011144701379950587, 0.02201340014536987, 0.018512094254542283]

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].bar(x - width/2, bsdr_fod, width, color='orange', label='BSDR')
axs[0].bar(x + width/2, ad_fod, width, color='blue', label='AD')
axs[0].set_ylabel('Mean Absolute FOD')
axs[0].set_title('First-Order Derivative (FOD)')
axs[0].legend()

axs[1].bar(x - width/2, bsdr_sod, width, color='orange', label='BSDR')
axs[1].bar(x + width/2, ad_sod, width, color='blue', label='AD')
axs[1].set_ylabel('Smoothness Index (SOD)')
axs[1].set_title('Smoothness of Retrieved Emissivity (SOD)')
axs[1].set_xticks(x)
axs[1].set_xticklabels(targets)
axs[1].set_xlabel('Target Size')
axs[1].legend()

plt.tight_layout()
plt.show()
