import numpy as np
printImportWarnings = False

regxASCII = "[^\W\d_]+"
plotlystrftime = '%Y-%m-%d %X' # for plotly timestamps


try:
    import h5py
except Exception as e:
    print('WARNING: ',e)
import numpy as np

def h5_nullterm_dtype(np_dtype: np.dtype) -> h5py.Datatype:
    dt = np.dtype(np_dtype)
    if dt.names is None:
        return dt  # Not a structured dtype
        # raise TypeError("Expected a structured dtype (with .names)")

    ct = h5py.h5t.create(h5py.h5t.COMPOUND, dt.itemsize)

    for name in dt.names:
        name_b = name.encode("utf-8") if isinstance(name, str) else name
        fdt, offset = dt.fields[name]

        if fdt.kind == "S":
            nbytes = fdt.itemsize
            st = h5py.h5t.C_S1.copy()
            st.set_size(nbytes)
            st.set_strpad(h5py.h5t.STR_NULLTERM)
            ct.insert(name_b, offset, st)
        else:
            ct.insert(name_b, offset, h5py.h5t.py_create(fdt, logical=True))

    return h5py.Datatype(ct)
def _bytes_dtype(n):
    return np.dtype(f"|S{max(int(n), 1)}")

def _bytes_array(items):
    if not items:
        return np.array([], dtype="|S1")
    bs = [bytes(x) if isinstance(x, (bytes, np.bytes_)) else str(x).encode("utf-8") for x in items]
    return np.array(bs, dtype=_bytes_dtype(max(len(b) for b in bs)))

def _as_bytes_value(v):
    if isinstance(v, (bytes, np.bytes_)):
        b = bytes(v)
        return b, _bytes_dtype(len(b))
    if isinstance(v, (str, np.str_)):
        b = str(v).encode("utf-8")
        return b, _bytes_dtype(len(b))
    if isinstance(v, np.ndarray):
        if v.dtype.kind in ("U", "S"):
            if v.shape == ():
                return _as_bytes_value(v.tolist())
            return (v, v.dtype) if v.dtype.kind == "S" else (lambda a: (a, a.dtype))(_bytes_array(v.tolist()))
        if v.dtype.kind == "O":
            if v.shape == ():
                return _as_bytes_value(v.tolist())
            items = v.tolist()
            if not items:
                a = np.array([], dtype="|S1")
                return a, a.dtype
            if all(isinstance(x, (str, np.str_, bytes, np.bytes_)) for x in items):
                a = _bytes_array(items)
                return a, a.dtype
    if isinstance(v, (list, tuple)):
        if not v:
            a = np.array([], dtype="|S1")
            return a, a.dtype
        if all(isinstance(x, (str, np.str_, bytes, np.bytes_)) for x in v):
            a = _bytes_array(v)
            return a, a.dtype
    return None, None

_attrs_init_done = False

def _init_attrs_bytes():
    global _attrs_init_done
    if _attrs_init_done:
        return
    _attrs_init_done = True
    try:
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        import funkshuns.validate_hdf as validate_hdf
        orig = validate_hdf._dcpl_sig
        def _dcpl_sig(ds, _orig=orig):
            sig = _orig(ds)
            fv = sig.get("fillvalue")
            if isinstance(fv, float) and np.isnan(fv):
                sig["fillvalue"] = None
            return sig
        validate_hdf._dcpl_sig = _dcpl_sig
    except Exception:
        pass

def set_attrs_bytes(dset, attrz):
    _init_attrs_bytes()
    for k, v in attrz.items():
        # remove old attr to allow dtype change
        if k in dset.attrs:
            del dset.attrs[k]
        bv, bdtype = _as_bytes_value(v)
        if bdtype is None:
            dset.attrs.create(k, v)
        else:
            dset.attrs.create(k, bv, dtype=bdtype)