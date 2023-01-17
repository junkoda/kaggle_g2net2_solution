import h5py


def load_flat_real(output_dir):
    """
    Returns:
      array[bool]: True for real-noise test data
    """
    with h5py.File(output_dir + '/test_real_noise.h5', 'r') as f:
        idx_real = f['idx'][:]

    return idx_real
