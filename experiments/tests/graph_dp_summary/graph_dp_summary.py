import ios
import time
import pandas as pd
import ios.optimizer
import ios.models.randwire
from ios.models import inception_v3
import faulthandler

faulthandler.enable()

graph = ios.models.randwire.randwire_xlarge()


for max_part_size in [30, 40, 45, 50, 55]:
    print(f"max_part_size = {max_part_size}")
    t1 = time.time()
    dp_info = ios.optimizer.graph_dp_summary(graph, max_part_size=max_part_size, max_num_groups=20, max_group_size=1)
    t2 = time.time()

    nparts = len(dp_info['width'])

    feat_names = ['width', '#states', '#transitions', '#schedules', '#operators']
    feat_types = [int, int, int, float, int]

    df = pd.DataFrame(columns=['part index'] + feat_names)

    for ipart in range(nparts):
        df.loc[len(df)] = [ipart] + [feat_type(dp_info[feat_name][ipart]) for feat_type, feat_name in zip(feat_types, feat_names)]
    
    print(df.to_string(index=False))
    print(f"Spent {t2-t1:.0f} secs")
    print()
