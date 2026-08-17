
import os
print("CKPTS=" + ",".join(sorted(
    f for f in os.listdir('/content/tuneA_root/weights') if f.startswith('0817-tune_wo_writing_from_rerun_s1000_'))))
