import numpy as np

def split_train_test(trajectories, train_frac=0.8, seed=0):
    rng = np.random.default_rng(seed)

    # bucket trajectories by their params (dictionary converted to a tuple key)
    buckets = {}
    for traj in trajectories:
        params = traj.get("params")
        key = tuple(sorted(params.items())) if params else None
        buckets.setdefault(key, []).append(traj)

    keys = list(buckets.keys())
    rng.shuffle(keys)

    if len(keys) > 1:
        split_idx = max(1, int(train_frac * len(keys)))
        split_idx = min(split_idx, len(keys) - 1)
        train_keys = set(keys[:split_idx])
        train = [traj for k in train_keys for traj in buckets[k]]
        test = [traj for k in keys[split_idx:] for traj in buckets[k]]
    else:
        pool = buckets[keys[0]]
        rng.shuffle(pool)
        split_idx = max(1, int(train_frac * len(pool)))
        split_idx = min(split_idx, len(pool) - 1)
        train = pool[:split_idx]
        test = pool[split_idx:]

    return train, test
