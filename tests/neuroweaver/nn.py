import numpy as np
import polymath as pm

from pathlib import Path
CWD = Path(f"{__file__}").parent
import locale
FLIST_FILE = "filelist_4"
REC_LENGTH = 49
REC_WINDOW = 10
LATITUDE_POS = 28
OPEN = 10000

def np_nn(input_info, file_lengths):
    out_info = {}

    out_info['z'] = np.zeros((REC_WINDOW,))
    target_lat, target_long = input_info['target_lat'], input_info['target_long']
    out_info['entries'] = input_info['entries'].copy()
    out_info['ndists'] = input_info['ndists'].copy()

    start = 0
    input_info['lats'] = np.empty(file_lengths[-1][-1], dtype=np.float64)
    input_info['longs'] = np.empty(file_lengths[-1][-1], dtype=np.float64)
    out_info['first'] = input_info['ndists'].copy()
    while (start + REC_WINDOW) < file_lengths[-1][-1]:

        sandbox = input_info['sandbox'][start: start + REC_WINDOW]

        orig_data = input_info['orig_data'][start: start + REC_WINDOW]
        start += REC_WINDOW


        for i in range(REC_WINDOW):
            if (start - REC_WINDOW) == 0:
                input_info['lats'][i] = sandbox[i][0]
                input_info['longs'][i] = sandbox[i][1]
            tmp_lat = sandbox[i][0]
            tmp_long = sandbox[i][1]
            out_info['z'][i] = np.sqrt(( (tmp_lat-target_lat) * (tmp_lat-target_lat) )+( (tmp_long-target_long) * (tmp_long-target_long)))

        for i in range(REC_WINDOW):
            max_dist = -1
            max_idx = 0
            for j in range(input_info['num']):
                if out_info['ndists'][j] > max_dist:
                    max_dist = out_info['ndists'][j]
                    max_idx = j
    #
            if out_info['z'][i] < out_info['ndists'][max_idx]:
    #
                out_info['entries'][max_idx] = orig_data[i]
                out_info['ndists'][max_idx] = out_info['z'][i]
                if (start - REC_WINDOW) == 0:
                    out_info['first'][max_idx] = out_info['z'][i]
                    # print(f"Here: {out_info['ndists']}\n")

    #
    input_info['sandbox'] = input_info['sandbox'][0:REC_WINDOW]
    input_info['orig_data'] = input_info['orig_data'][0:REC_WINDOW]
    input_info['longs'] = input_info['longs'][0:REC_WINDOW]
    input_info['lats'] = input_info['lats'][0:REC_WINDOW]
    input_info.pop("sandbox")
    # for j in range(input_info['num']):
    #     if out_info['ndists'][j] != OPEN:
    #         print(f"{out_info['entries'][j]} --> {out_info['ndists'][j]}")
    return out_info

def read_filelist(listfile):
    with open(f"{listfile}", "r") as f:
        lines = f.readlines()
    lines = [f"{CWD}/data/nn/{Path(l.strip()).name}" for l in lines]
    return lines

def nn_datagen(flist_file, neighbors, tgt_lat, tgt_long, rec_len, rec_window, lowered=False, inf=False):
    full_paths = read_filelist(flist_file)
    input_info = {}
    data = []
    file_lengths = []
    start = 0
    for fname in full_paths:
        with open(fname, "r") as f:
            fdata = f.readlines()
            data += fdata
            file_lengths.append((start, start+len(fdata)))
            start = start+len(fdata) + 1
    assert all([len(d) == rec_len for d in data])
    input_info['orig_data'] = data
    np_data = np.empty((len(data), 2), dtype=np.float64)
    for i, d in enumerate(data):
        lat = float(d[LATITUDE_POS-1:LATITUDE_POS+4])
        long = float(d[LATITUDE_POS+5:LATITUDE_POS+10])
        np_data[i][0] = lat
        np_data[i][1] = long

    input_info['sandbox'] = np_data
    input_info['target_lat'] = tgt_lat
    input_info['target_long'] = tgt_long
    input_info['num'] = neighbors
    input_info['entries'] = np.full((neighbors), '', dtype=np.dtype('<U48'))

    input_info['ndists'] = np.full(neighbors, OPEN, dtype=np.float64)

    out_info = np_nn(input_info, file_lengths)

    input_info.pop("entries")
    input_info.pop("orig_data")
    input_info.pop("num")

    if lowered:
        all_keys = []
        for i in range(rec_window):
            ndist_key = f"ndists/ndists({i},)"
            all_keys.append(ndist_key)
            for j in range(rec_len):
                entry_key = f"entries/entries({i}, {j})"
                all_keys.append(entry_key)
                input_info[entry_key] = input_info["entries"][i,j]
            input_info[ndist_key] = input_info["ndists"][i]
        input_info.pop("entries")
        input_info.pop("ndists")
    else:
        all_keys = ["ndists", "entries"]


    return input_info, all_keys, out_info

def nn_impl_(neighbors, latitude,longitude,  coarse=False):
    with pm.Node(name="nn") as graph:
        # num = neighbors
        num = pm.parameter("num")
        target_lat = pm.parameter("target_lat")
        target_long = pm.parameter("target_long")

        lats = pm.input("lats", shape=(REC_WINDOW))
        longs = pm.input("longs", shape=(REC_WINDOW))

        # Initialize to OPEN
        # ndists = pm.state("ndists", shape=(num))
        # ndists_temp = pm.state("ndists_temp", shape=(REC_WINDOW,))
        ndists = pm.state("ndists", shape=(REC_WINDOW,))
        # ndists = pm.output("ndists", shape=(REC_WINDOW,))
        # entries = pm.output("entries", shape=(num))

        i = pm.index(0, REC_WINDOW - 1, name="i")
        j = pm.index(0, num - 1, name="j")
        # tmp_lats = sandbox[i, 0]
        # tmp_longs = sandbox[i, 1]

        z = pm.sqrt(((lats[i]-target_lat) * (lats[i]-target_lat)) + ((longs[i]-target_long) * (longs[i]-target_long)), name="sqrtz")
        max_dist = pm.sum([i], ndists[i], name=f"max_dist")
        min_dist = pm.sum([i], ndists[i], name=f"min_dist")
        # Uncommenting this line generates this error:
        # ndists[j] = (z[j] > min_dist) * z[j] + (z[j] <= max_dist) * z[j]
        # ndists_temp[i] = (z[i] > min_dist) * z[i] + (z[i] <= max_dist) * z[i]
        ndists_temp = (z[i] > min_dist) * z[i] + (z[i] <= max_dist) * z[i]
        # Add a forloop around this computation, to get cycles for several iterations
        for idx in range(1, REC_WINDOW):
            max_dist = pm.sum([i], ndists_temp[i], name=f"max_dist{idx}")
            min_dist = pm.sum([i], ndists_temp[i], name=f"min_dist{idx}")
        # Uncommenting this line generates this error:
        # ndists[j] = (z[j] > min_dist) * z[j] + (z[j] <= max_dist) * z[j]
        # ndists_temp[i] = (z[i] > min_dist) * z[i] + (z[i] <= max_dist) * z[i]
            ndists_temp = (z[i] > min_dist) * z[i] + (z[i] <= max_dist) * z[i]
        '''
        ndists[j] = (z[j] > min_dist)*z[j] + (z[j] <= max_dist)*z[j]
                if state_name not in self.possible_states:
>           raise ValueError(f"{state_name} is not a possible state for "
                             f"{self.component_type}.")
E           ValueError: lt is not a possible state for pe.
        '''
        # delta = (z[i] + max_dist).set_name(f"delta_set")
        ndists[i] = (ndists_temp[i] + ndists[i]).set_name("ndists_set")

    if coarse:
        in_info, keys, out_info = nn_datagen(f"{CWD}/openmp/nn/filelist_4", neighbors, latitude, longitude,
                                                    REC_LENGTH, REC_WINDOW)

        return graph, in_info, out_info, keys
    else:
        in_info, keys, out_info = nn_datagen(f"{CWD}/openmp/nn/filelist_4", neighbors, latitude, longitude,
                                                    REC_LENGTH, REC_WINDOW, lowered=True)
        shape_val_pass = pm.NormalizeGraph({"num": in_info['num']})
        new_graph = shape_val_pass(graph)
        return new_graph, in_info, out_info, keys

def nn_impl(neighbors, latitude,longitude,  coarse=False):

    with pm.Node(name="nn") as graph:
        num = pm.parameter("num")
        target_lat = pm.parameter("target_lat")
        target_long = pm.parameter("target_long")
        # target_lat = latitude
        # target_long = longitude

        lats = pm.input("lats", shape=(REC_WINDOW,))
        longs = pm.input("longs", shape=(REC_WINDOW,))
        # z = pm.state("z", shape=(REC_WINDOW,))

        # Initialize to OPEN
        ndists = pm.state("ndists", shape=(num,))

        i = pm.index(0, REC_WINDOW - 1, name="i")
        j = pm.index(0, num - 1, name="j")

        t = (((lats[i]-target_lat) * (lats[i]-target_lat)) + ((longs[i]-target_long) * (longs[i]-target_long))).set_name("tempz")
        # t = (lats[i]-target_lat).set_name("tempz")
        z = pm.sqrt(t, name="sqrtz")
        # z = pm.sqrt(((lats[i]-target_lat) * (lats[i]-target_lat)) + ((longs[i]-target_long) * (longs[i]-target_long)), name="sqrtz")


        # for idx in range(REC_WINDOW):
        idx = pm.index(0, 1, name="idx")
        # idx = 0
        max_dist = pm.max([j], ndists[j], name=f"max_dist{idx}")
            # max_idx = (pm.argmax([j], ndists[j])).set_name(f"max_idx{idx}")
        # ndists[0] = (((z[idx] < max_dist) * z[idx]) + ((z[idx] >= max_dist) * ndists[0]))
        ndists[j] = ((z[j]) * max_dist).set_name("ndist_write")
            # ndists[0] = ((t[idx]) * max_dist)

    if coarse:
        in_info, keys, out_info = nn_datagen(f"{CWD}/openmp/nn/filelist_4", neighbors, latitude, longitude,
                                                    REC_LENGTH, REC_WINDOW)

        return graph, in_info, out_info, keys
    else:
        in_info, keys, out_info = nn_datagen(f"{CWD}/openmp/nn/filelist_4", neighbors, latitude, longitude,
                                                    REC_LENGTH, REC_WINDOW, lowered=True)
        shape_val_pass = pm.NormalizeGraph({"num": in_info['num']})
        new_graph = shape_val_pass(graph)
        return new_graph, in_info, out_info, keys