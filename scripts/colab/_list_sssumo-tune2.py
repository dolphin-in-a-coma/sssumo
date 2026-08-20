
import os
print("CKPTS=" + ",".join(sorted(
    f for f in os.listdir('/content/tune2_root/weights') if f.startswith('0817-tune_wo_writing_s1000_'))))
