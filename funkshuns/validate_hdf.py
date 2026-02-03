import h5py
import numpy as np

# --- helpers to pretty-print HDF5 low-level enums ---
_STRPAD = {
    h5py.h5t.STR_NULLTERM: "null-terminated",
    h5py.h5t.STR_NULLPAD: "null-padded",
    h5py.h5t.STR_SPACEPAD: "space-padded",
}
_ORDER = {
    h5py.h5t.ORDER_LE: "little-endian",
    h5py.h5t.ORDER_BE: "big-endian",
    h5py.h5t.ORDER_VAX: "vax",
    h5py.h5t.ORDER_NONE: "none",
}
_SIGN = {
    h5py.h5t.SGN_NONE: "unsigned",
    h5py.h5t.SGN_2: "signed",
}

def _describe_member(mt):
    """Describe a member (TypeID) of a compound dtype at the HDF5 level."""
    cls = mt.get_class()
    info = {
        "class": {h5py.h5t.INTEGER: "Integer",
                  h5py.h5t.FLOAT: "Float",
                  h5py.h5t.STRING: "String",
                  h5py.h5t.COMPOUND: "Compound",
                  h5py.h5t.ENUM: "Enum",
                  h5py.h5t.BITFIELD: "Bitfield",
                  h5py.h5t.OPAQUE: "Opaque",
                  h5py.h5t.REFERENCE: "Reference",
                  h5py.h5t.VLEN: "Vlen",
                  h5py.h5t.ARRAY: "Array"}.get(cls, f"Class{cls}"),
        "size": mt.get_size(),
    }
    # Only numeric types have endianness
    if cls in (h5py.h5t.INTEGER, h5py.h5t.FLOAT):
        info["endianness"] = _ORDER.get(mt.get_order(), str(mt.get_order()))
    if cls == h5py.h5t.INTEGER:
        info["signed"] = _SIGN.get(mt.get_sign(), str(mt.get_sign()))
    elif cls == h5py.h5t.FLOAT:
        # floats have no sign; size+order already reported
        pass
    elif cls == h5py.h5t.STRING:
        # Determine fixed vs vlen and string padding/charset
        try:
            is_vlen = h5py.h5t.is_variable_str(mt)
        except Exception:
            is_vlen = False
        info["vlen"] = bool(is_vlen)
        info["charSet"] = "UTF-8" if mt.get_cset() == h5py.h5t.CSET_UTF8 else "ASCII"
        try:
            info["strPad"] = _STRPAD.get(mt.get_strpad(), str(mt.get_strpad()))
        except Exception:
            info["strPad"] = "unknown"
    return info

def _describe_dataset(ds):
    """
    Return a full low-level description of an h5py.Dataset's compound dtype and creation layout.
    """
    t = ds.id.get_type()
    is_compound = (t.get_class() == h5py.h5t.COMPOUND)
    desc = {
        "shape": tuple(ds.shape),
        "chunks": ds.chunks,
        "compression": ds.compression,
        "compression_opts": ds.compression_opts,
        "shuffle": ds.shuffle,
        "fletcher32": ds.fletcher32,
        "dtype": str(ds.dtype),
        "itemsize": ds.dtype.itemsize,
        "compound": is_compound,
        "nmembers": t.get_nmembers() if is_compound else 0,
        "members": [],
    }
    if is_compound:
        nm = t.get_nmembers()
        for i in range(nm):
            name = t.get_member_name(i)
            # h5py may return bytes in some versions
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            mt = t.get_member_type(i)
            mem = _describe_member(mt)
            mem.update({
                "name": name,
                "offset": t.get_member_offset(i),
            })
            desc["members"].append(mem)
    return desc

import numpy as np

def _compare_scalar(a, b, label, diffs):
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return
    if a != b:
        diffs.append({"field": label, "new": a, "ref": b})


def _compare_dict(new, ref, prefix, diffs):
    """recursively compare two dicts; any difference is appended to diffs"""
    all_keys = set(new.keys()) | set(ref.keys())
    for k in sorted(all_keys):
        label = f"{prefix}.{k}"
        if k not in ref:
            diffs.append({"field": label, "new": new[k], "ref": "(missing)"})
        elif k not in new:
            diffs.append({"field": label, "new": "(missing)", "ref": ref[k]})
        else:
            nv, rv = new[k], ref[k]
            if isinstance(nv, dict) and isinstance(rv, dict):
                _compare_dict(nv, rv, label, diffs)
            elif nv != rv:
                diffs.append({"field": label, "new": nv, "ref": rv})

def validate_like(ds, like, *, check_layout=True, strict_order=True, require_same_shape=False):
    """
    Validate that dataset `ds` matches the schema/layout of dataset `like`.

    Checks:
      • shape (optionally requiring exact match)
      • compound/non-compound
      • number of members (if compound)
      • member names, classes, sizes, offsets (if compound)
      • optionally chunk size, compression, shuffle, fletcher32 (if check_layout=True)
      • if strict_order=True, members must appear in the same order by offset

    Returns (bool, dict):
        ok, report = validate_like(...)
        if not ok:
            print(report["summary"])
            for diff in report["diffs"]:
                print(f"  {diff['field']}: {diff['new']} vs {diff['ref']}")
    
    Usage:
        ref = hdf["/Modifications/DetentionBasin_WI01/Attributes"]   # go-by
        new = hdf["/Modifications/DetentionBasin_WI02/Attributes"]   # your output
        ok, report = validate_like(new, ref, check_layout=True, strict_order=True, require_same_shape=False)
        if not ok:
            print("❌ Schema/layout mismatch:")
            print(report["summary"])
            for diff in report["diffs"]:
                print(f"  • {diff['field']}: got {diff['new']}, expected {diff['ref']}")
        else:
            print("✅ Schema matches perfectly.")

    Example output dict if there are issues:
        {
            "ok": False,
            "summary": "Found 3 difference(s) in schema/layout",
            "diffs": [
                {"field": "members[1].name", "new": "StorageCapacity", "ref": "storageCapacity"},
                {"field": "members[2].class", "new": "Integer", "ref": "Float"},
                {"field": "chunks", "new": (50,), "ref": (100,)}
            ]
        }
    """
    ref = _describe_dataset(like)
    new = _describe_dataset(ds)
    diffs = []
    
    # Compare top-level shape
    if require_same_shape:
        _compare_scalar(new["shape"], ref["shape"], "shape", diffs)
    
    # Compare compound vs non-compound
    _compare_scalar(new["compound"], ref["compound"], "compound", diffs)
    
    # Optionally compare layout/creation properties
    if check_layout:
        _compare_scalar(new["chunks"], ref["chunks"], "chunks", diffs)
        _compare_scalar(new["compression"], ref["compression"], "compression", diffs)
        _compare_scalar(new["compression_opts"], ref["compression_opts"], "compression_opts", diffs)
        _compare_scalar(new["shuffle"], ref["shuffle"], "shuffle", diffs)
        _compare_scalar(new["fletcher32"], ref["fletcher32"], "fletcher32", diffs)
    
    # If compound, dig into the member definitions
    if ref["compound"]:
        _compare_scalar(new["nmembers"], ref["nmembers"], "nmembers", diffs)
        if new["nmembers"] == ref["nmembers"]:
            # Check member by member
            for i, (nm, rm) in enumerate(zip(new["members"], ref["members"])):
                prefix = f"members[{i}]"
                _compare_dict(nm, rm, prefix, diffs)
            
            # Optionally check offset ordering
            if strict_order:
                new_order = [m["offset"] for m in new["members"]]
                ref_order = [m["offset"] for m in ref["members"]]
                if new_order != ref_order:
                    diffs.append({
                        "field": "member_offset_order",
                        "new": new_order,
                        "ref": ref_order
                    })
    
    ok = len(diffs) == 0
    report = {
        "ok": ok,
        "summary": "Schema matches perfectly." if ok else f"Found {len(diffs)} difference(s) in schema/layout",
        "diffs": diffs,
    }
    return ok, report

_LAYOUT = {
    h5py.h5d.COMPACT: "compact",
    h5py.h5d.CONTIGUOUS: "contiguous",
    h5py.h5d.CHUNKED: "chunked",
    h5py.h5d.VIRTUAL: "virtual",
}

def _obj_kind(obj):
    if isinstance(obj, h5py.Dataset):
        return "dataset"
    if isinstance(obj, h5py.Group):
        return "group"
    return type(obj).__name__

def _attrs_sig(obj):
    # small, stable signature: keys + (type, shape) + value bytes for small scalars
    sig = {}
    for k in obj.attrs.keys():
        v = obj.attrs[k]
        # normalize numpy scalars/arrays for compare
        if isinstance(v, np.ndarray):
            sig[k] = ("ndarray", v.dtype.str, tuple(v.shape), v.tobytes() if v.size <= 64 else None)
        else:
            vv = np.array(v)
            sig[k] = ("scalar", vv.dtype.str, tuple(vv.shape), vv.tobytes() if vv.size <= 64 else None)
    return sig

def _dcpl_sig(ds: h5py.Dataset):
    dcpl = ds.id.get_create_plist()
    sig = {
        "layout": _LAYOUT.get(dcpl.get_layout(), str(dcpl.get_layout())),
        "chunks": ds.chunks,
        "compression": ds.compression,
        "compression_opts": ds.compression_opts,
        "shuffle": ds.shuffle,
        "fletcher32": ds.fletcher32,
        "fill_time": getattr(dcpl, "get_fill_time", lambda: None)(),
        "track_times": getattr(dcpl, "get_obj_track_times", lambda: None)(),
        "attr_creation_order": getattr(dcpl, "get_attr_creation_order", lambda: None)(),
    }
    # fillvalue is tricky; only include if h5py reports it
    try:
        sig["fillvalue"] = ds.fillvalue
    except Exception:
        sig["fillvalue"] = None
    return sig

def _link_sig(h5: h5py.File, path: str):
    # detect if something is a soft/external link
    try:
        li = h5py.h5l.get_info(h5.id, path.encode("utf-8"))
        t = li.type
        if t == h5py.h5l.H5L_TYPE_HARD:
            return ("hard", None)
        if t == h5py.h5l.H5L_TYPE_SOFT:
            return ("soft", h5py.h5l.get_val(h5.id, path.encode("utf-8")))
        if t == h5py.h5l.H5L_TYPE_EXTERNAL:
            return ("external", h5py.h5l.get_val(h5.id, path.encode("utf-8")))
        return (f"type_{t}", None)
    except Exception:
        return ("unknown", None)

def validd2(demhdf0, demhdf, base="/Modifications/DetentionBasin_WI01", *, check_attrs=True, check_dcpl=True):
    with h5py.File(demhdf0, "r") as h0, h5py.File(demhdf, "r") as h1:
        if base not in h0 or base not in h1:
            print("X base missing:", base, "ref?", base in h0, "new?", base in h1)
            return

        # collect all object paths under base (datasets + groups)
        ref_paths, new_paths = set(), set()

        def _collect(h, paths):
            def _cb(name, obj):
                paths.add(f"{base}/{name}" if name else base)
            h[base].visititems(_cb)

        _collect(h0, ref_paths)
        _collect(h1, new_paths)

        missing = sorted(ref_paths - new_paths)
        extra = sorted(new_paths - ref_paths)

        if missing:
            print("X Missing objects:")
            for p in missing:
                print(" -", p)
        if extra:
            print("! Extra objects:")
            for p in extra:
                print(" -", p)

        # compare common paths
        diffs = []
        for p in sorted(ref_paths & new_paths):
            o0, o1 = h0[p], h1[p]

            # link type compare (rare but matters)
            l0, l1 = _link_sig(h0, p), _link_sig(h1, p)
            if l0 != l1:
                diffs.append(f"{p}: link differs new={l1} ref={l0}")

            # type compare
            k0, k1 = _obj_kind(o0), _obj_kind(o1)
            if k0 != k1:
                diffs.append(f"{p}: object type differs new={k1} ref={k0}")
                continue

            # group attrs compare
            if check_attrs and isinstance(o0, h5py.Group):
                a0, a1 = _attrs_sig(o0), _attrs_sig(o1)
                if a0 != a1:
                    diffs.append(f"{p}: group attrs differ")

            # dataset-level checks
            if isinstance(o0, h5py.Dataset):
                # existing low-level schema/layout checker
                ok, rep = validate_like(o1, o0, check_layout=True, strict_order=True, require_same_shape=False)
                if not ok:
                    diffs.append(f"{p}: schema/layout mismatch ({len(rep['diffs'])} diffs)")
                    for d in rep["diffs"][:6]:
                        diffs.append(f"  - {d}")

                if check_dcpl:
                    d0, d1 = _dcpl_sig(o0), _dcpl_sig(o1)
                    if d0 != d1:
                        diffs.append(f"{p}: dcpl/layout props differ")
                        for k in sorted(set(d0) | set(d1)):
                            if d0.get(k) != d1.get(k):
                                diffs.append(f"  - {k}: new={d1.get(k)} ref={d0.get(k)}")

                if check_attrs:
                    a0, a1 = _attrs_sig(o0), _attrs_sig(o1)
                    if a0 != a1:
                        diffs.append(f"{p}: dataset attrs differ")
                        keys = sorted(set(a0) | set(a1))
                        for k in keys:
                            if a0.get(k) != a1.get(k):
                                diffs.append(f"  - attr {k}: new={a1.get(k)} ref={a0.get(k)}")

        if diffs:
            print("\nX Differences found:")
            for d in diffs:
                print(d)
        else:
            print("OK Entire subtree matches (schema/layout + attrs + dcpl).")
        return diffs

def validd(demhdf0, demhdf,base):
    hdf0 = h5py.File(demhdf0, "r")
    hdf = h5py.File(demhdf, "r")
    tblnms = list(dict(hdf[f"{base}"]).keys())
    for tblnm in tblnms:
        print(tblnm)
        ref = hdf0[f"{base}/{tblnm}"]   # go-by
        new =  hdf[f"{base}/{tblnm}"]                  # your output

        ok, report = validate_like(new, ref, check_layout=True, strict_order=True, require_same_shape=False)
        if not ok:
            print("❌ Schema/layout mismatch:")
            for d in report["diffs"]:
                print(" -", d)
        else:
            print("✅ Dataset matches reference schema/layout.")
    hdf0.close()
    hdf.close()

def validate_hdf(refhdf,newhdf,base,strict=True):
    '''base: path to group to validate under'''
    if strict:
        return validd2(refhdf,newhdf,base)
    else:
        return validd(refhdf,newhdf,base)