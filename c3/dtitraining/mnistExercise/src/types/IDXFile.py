import numpy as np
import io
import gzip
import requests

def url_to_filepath(url):
    return url.replace('/', '_').replace(':', '_')

def check_for_existing_spec(init_path):
    # Build fetch spec and fetch
    fetch_spec = c3.FetchSpec(
        filter="startsWith(location, '{}')".format(init_path),
        include="id, location")
    file_sources_from_fetch = c3.FileSourceSpec.fetch(spec=fetch_spec)
    # Check if there's results
    if file_sources_from_fetch.objs is not None:
        if len(file_sources_from_fetch.objs) > 1:
            print("WARNING: Multiple compatible file sources exist for url {}".format(this.url))
        source_id = file_sources_from_fetch.objs[0].id
        return c3.FileSourceSpec.get(source_id)
    else:
        return None

def numpy_from_idx(infile):
    bytecode_type_map = {
        0x08: np.ubyte,
        0x09: np.byte,
        0x0B: np.short,
        0x0C: np.intc,
        0x0D: np.single,
        0x0E: np.double,
    }

    # Open downloaded file using memory buffer.
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

    return data.reshape(dimensions)

def getFileSourceSpec(this, enableLocalClientStorage):
    filepath = url_to_filepath(this.url)

    # First, check whether the file already exists
    spec = check_for_existing_spec(filepath)
    if spec is not None:
        return spec

    # Download IDX file
    r = requests.get(this.url) 

    # Open downloaded file using memory buffer.
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as infile:
        data = numpy_from_idx(infile)

    spec = c3.FileSourceCreateSpec(
        locationPrefix="{}".format(filepath),
        enableLocalClientStorage=enableLocalClientStorage)
    return c3.FileSourceSpec.createFromNumpy(data, spec)

def getFileSourceSpecPreprocess(this, serializedPreprocessor, preprocessFuncName, enableLocalClientStorage):
    # Unpickle the preprocessor
    preprocessor = c3.PythonSerialization.deserialize(serializedPreprocessor)

    filepath = '-'.join([preprocessFuncName, url_to_filepath(this.url)])

    # Check whether the file already exists
    spec = check_for_existing_spec(filepath)
    if spec is not None:
        return spec

    # Download IDX file
    r = requests.get(this.url)

    # Open downloaded file using memory buffer.
    with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as infile:
        data = numpy_from_idx(infile)

    preprocessed_data = preprocessor(data)
    if type(preprocessed_data) is not np.ndarray:
        raise RuntimeError("ERROR: Preprocessing function must return a numpy array!")

    # Store numpy data as file spec
    spec = c3.FileSourceCreateSpec(
        locationPrefix="{}".format(filepath),
        enableLocalClientStorage=enableLocalClientStorage)
    return c3.FileSourceSpec.createFromNumpy(preprocessed_data, spec)
