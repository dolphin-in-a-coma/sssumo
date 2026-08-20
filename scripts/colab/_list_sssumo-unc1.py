
import os
print("CKPTS=" + ",".join(sorted(
    f for f in os.listdir('/content/pertrial_unc') if f.startswith('organic.'))))
