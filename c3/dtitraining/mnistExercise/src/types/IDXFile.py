import numpy as np
import io
import gzip
import requests

def getDataset(this):
    bytecode_type_map = {
        0x08: np.ubyte,
        0x09: np.byte,
        0x0B: np.short,
        0x0C: np.intc,
        0x0D: np.single,
        0x0E: np.double,
    }

    # Download IDX file
    r = requests.get(this.url) 

    # Open downloaded file using memory buffer.
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as infile:
        if int.from_bytes(infile.read(2), 'big') != 0:
            raise RuntimeError("Improperly formatted IDX file. First two bytes should be 0.")

        data_type = int.from_bytes(infile.read(1), 'big')
        num_dimensions = int.from_bytes(infile.read(1), 'big')
        dimensions = []
        for i in range(num_dimensions):
            dimensions.append(int.from_bytes(infile.read(4), 'big'))

        total_len = 1
        for dim_len in dimensions:
            total_len *= dim_len

        itemsize = np.dtype(bytecode_type_map[data_type]).itemsize
        data = np.frombuffer(infile.read(itemsize*total_len), dtype=bytecode_type_map[data_type])
        data = data.reshape(dimensions)

    return c3.Dataset.fromPython(pythonData=data)
