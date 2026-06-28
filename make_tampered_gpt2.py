#!/usr/bin/env python3
"""Derive a tampered gpt2 from a clean safetensors file by scaling exactly one
MLP tensor in one layer, leaving every other tensor byte-identical.

Pure standard library (no torch / numpy / safetensors needed) — it edits the
raw F32 bytes in place. Used to produce a fixture for `assay compare`:

    python make_tampered_gpt2.py ./models/gpt2/model.safetensors \
        ./models/gpt2-tampered/model.safetensors --layer 5 --scale 4.0
"""
import argparse
import json
import os
import struct
import sys


def find_tensor(header, layer):
    # gpt2 MLP up-projection, with or without the `transformer.` prefix.
    for name in (
        f"h.{layer}.mlp.c_fc.weight",
        f"transformer.h.{layer}.mlp.c_fc.weight",
        f"model.layers.{layer}.mlp.c_fc.weight",
    ):
        if name in header:
            return name
    # Fallback: first tensor whose name mentions this layer and "mlp".
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        if f".{layer}." in name and "mlp" in name and "weight" in name:
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--scale", type=float, default=4.0)
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        blob = bytearray(f.read())

    header_len = struct.unpack_from("<Q", blob, 0)[0]
    header = json.loads(bytes(blob[8 : 8 + header_len]))
    data_start = 8 + header_len

    name = find_tensor(header, args.layer)
    if name is None:
        sys.exit(f"no MLP tensor found for layer {args.layer}")
    spec = header[name]
    if spec["dtype"] != "F32":
        sys.exit(f"tensor {name} is {spec['dtype']}, this script only scales F32")

    begin, end = spec["data_offsets"]
    off0, off1 = data_start + begin, data_start + end
    count = (end - begin) // 4
    vals = list(struct.unpack_from(f"<{count}f", blob, off0))
    vals = [v * args.scale for v in vals]
    struct.pack_into(f"<{count}f", blob, off0, *vals)

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    with open(args.dst, "wb") as f:
        f.write(blob)
    print(f"scaled '{name}' ({count} F32 weights) by {args.scale} -> {args.dst}")


if __name__ == "__main__":
    main()
