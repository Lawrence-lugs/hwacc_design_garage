from hwacctools.comp_graph import cgraph, cnodes, core
import onnx
import matplotlib.pyplot as plt
import rectpack
import seaborn as sns
import pandas as pd

def run_packer_sweeps(nx_model):
    # Experiments on all relevant packers
    pack_alg = {
        'BL'    :rectpack.MaxRectsBl,
        'BSSF'  :rectpack.MaxRectsBssf,
        'BAF'   :rectpack.MaxRectsBaf,
    }
    mode = {
        'ON'    :rectpack.PackingMode.Online,
        'OFF'   :rectpack.PackingMode.Offline,
    }
    sort_alg = { # Ratio and Sside are the best-performing sorts
        'SO'    :rectpack.SORT_NONE, 
        'SF'    :rectpack.SORT_RATIO,
        'SS'    :rectpack.SORT_SSIDE,
    }
    bin_alg = {
        'BNF'   :rectpack.PackingBin.BNF,
        'BBF'   :rectpack.PackingBin.BBF,
        'BFF'   :rectpack.PackingBin.BBF,
    }

    packer_dict = {
        f"MR-{m}-{p}-{b}-{s}": rectpack.newPacker(
            mode=mode[m],
            bin_algo=bin_alg[b],
            rotation=False,
            pack_algo=pack_alg[p],
            sort_algo=sort_alg[s],
        ) for m in mode for p in pack_alg for b in bin_alg for s in sort_alg
    }

    sim_df = pd.DataFrame(columns=['set_name','total_bins', 'total_bin_writes',])
    sim_df[['heuristic','mode', 'pack_algo', 'bin_algo', 'sort_algo']] = sim_df['set_name'].str.split('-', expand=True)
    # convert mode, pack_algo, bin_algo, sort_algo to categorical
    sim_df['mode'] = sim_df['mode'].astype('category')
    sim_df['pack_algo'] = sim_df['pack_algo'].astype('category')
    sim_df['bin_algo'] = sim_df['bin_algo'].astype('category')
    sim_df['sort_algo'] = sim_df['sort_algo'].astype('category')
    # convert total_bins and total_bin_writes to int
    sim_df['total_bins'] = sim_df['total_bins'].astype(int)
    sim_df['total_bin_writes'] = sim_df['total_bin_writes'].astype(int)
    return sim_df