
import os
print("CKPTS=" + ",".join(sorted(
    f for f in os.listdir('/content/tuneA_root/weights') if f.startswith('0817-ModGaussian_ampl_rerun_s1000_'))))
