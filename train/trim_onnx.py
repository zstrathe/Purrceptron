#!/usr/bin/env python3
"""Trim YOLOv8 ONNX to raw conv outputs for Hailo compilation.

Ultralytics exports the full Detect head (model.22) which includes Sigmoid
and DFL (depth_to_space/Reshape) after the raw conv layers. Hailo's
nms_postprocess expects the model to end at raw conv outputs.

This script trims the graph to end at the 6 detection head conv outputs:
  - 3 regression convs (cv2.0.2, cv2.1.2, cv2.2.2)
  - 3 classification convs (cv3.0.2, cv3.1.2, cv3.2.2)

Works for all YOLOv8 variants (n/s/m/l/x).

Usage:
    python3 trim_onnx.py <input.onnx> <output.onnx>
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict

import onnx
from onnx import TensorProto, helper, shape_inference


def find_detect_head_convs(graph: onnx.GraphProto) -> tuple[list[str], list[str]]:
    """Find the 6 Detect head conv output tensor names.

    Returns (reg_outputs, cls_outputs) where each list has 3 tensor names
    ordered by scale (stride 8, 16, 32).
    """
    # Primary: match by node name pattern
    # Ultralytics naming: /model.22/cv2.{0,1,2}/cv2.{0,1,2}.2/Conv (reg)
    #                     /model.22/cv3.{0,1,2}/cv3.{0,1,2}.2/Conv (cls)
    reg_pattern = re.compile(r"/model\.22/cv2\.(\d)/cv2\.\d\.2/Conv$")
    cls_pattern = re.compile(r"/model\.22/cv3\.(\d)/cv3\.\d\.2/Conv$")

    reg_outputs: dict[int, str] = {}
    cls_outputs: dict[int, str] = {}

    for node in graph.node:
        m = reg_pattern.search(node.name)
        if m and node.op_type == "Conv":
            scale_idx = int(m.group(1))
            reg_outputs[scale_idx] = node.output[0]
            continue
        m = cls_pattern.search(node.name)
        if m and node.op_type == "Conv":
            scale_idx = int(m.group(1))
            cls_outputs[scale_idx] = node.output[0]

    if len(reg_outputs) == 3 and len(cls_outputs) == 3:
        reg = [reg_outputs[i] for i in sorted(reg_outputs)]
        cls = [cls_outputs[i] for i in sorted(cls_outputs)]
        return reg, cls

    # Fallback: find Conv nodes feeding into Sigmoid (cls) or DFL chain (reg)
    print("[WARN] Primary pattern match failed, using fallback detection")
    consumer_map: dict[str, list[onnx.NodeProto]] = defaultdict(list)
    for node in graph.node:
        for inp in node.input:
            consumer_map[inp].append(node)

    reg_fallback = []
    cls_fallback = []

    for node in graph.node:
        if node.op_type != "Conv" or "/model.22/" not in node.name:
            continue
        output = node.output[0]
        consumers = consumer_map.get(output, [])
        if not consumers:
            continue
        if any(c.op_type == "Sigmoid" for c in consumers):
            cls_fallback.append(output)
        elif any(c.op_type in ("Reshape", "Transpose") for c in consumers):
            reg_fallback.append(output)

    if len(reg_fallback) == 3 and len(cls_fallback) == 3:
        return reg_fallback, cls_fallback

    # Debug: list all Conv nodes under model.22
    print("[ERROR] Could not find exactly 6 Detect head conv outputs")
    print(f"  Found {len(reg_fallback)} reg, {len(cls_fallback)} cls (fallback)")
    print("  Conv nodes under /model.22/:")
    for node in graph.node:
        if node.op_type == "Conv" and "/model.22/" in node.name:
            print(f"    {node.name} -> {node.output[0]}")
    sys.exit(1)


def get_tensor_shape(graph: onnx.GraphProto, tensor_name: str) -> list[int] | None:
    """Get shape of a tensor from value_info or graph outputs."""
    for vi in list(graph.value_info) + list(graph.output):
        if vi.name == tensor_name:
            dims = vi.type.tensor_type.shape.dim
            return [d.dim_value if d.dim_value > 0 else -1 for d in dims]
    return None


def trim_to_outputs(
    model: onnx.ModelProto, output_names: list[str]
) -> onnx.ModelProto:
    """Set new outputs and prune unreachable nodes."""
    graph = model.graph

    # Build output-to-node map for backward traversal
    output_to_node: dict[str, onnx.NodeProto] = {}
    for node in graph.node:
        for out in node.output:
            output_to_node[out] = node

    # Backward reachability from new outputs
    reachable_tensors: set[str] = set()
    stack = list(output_names)
    while stack:
        tensor = stack.pop()
        if tensor in reachable_tensors:
            continue
        reachable_tensors.add(tensor)
        if tensor in output_to_node:
            node = output_to_node[tensor]
            for inp in node.input:
                stack.append(inp)

    # Keep only reachable nodes
    reachable_nodes = []
    for node in graph.node:
        if any(out in reachable_tensors for out in node.output):
            reachable_nodes.append(node)

    # Create new output value_infos
    new_outputs = []
    for name in output_names:
        shape = get_tensor_shape(graph, name)
        if shape:
            new_outputs.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            )
        else:
            # Minimal value info without shape
            new_outputs.append(
                helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            )

    # Keep only reachable initializers
    reachable_initializers = [
        init for init in graph.initializer if init.name in reachable_tensors
    ]

    # Build new graph
    new_graph = helper.make_graph(
        reachable_nodes,
        graph.name,
        list(graph.input),
        new_outputs,
        initializer=reachable_initializers,
    )

    new_model = helper.make_model(new_graph, opset_imports=list(model.opset_import))
    new_model.ir_version = model.ir_version
    return new_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim YOLOv8 ONNX to raw conv outputs for Hailo"
    )
    parser.add_argument("input", help="Input ONNX model path")
    parser.add_argument("output", help="Output trimmed ONNX path")
    args = parser.parse_args()

    print(f"[INFO] Loading {args.input}")
    model = onnx.load(args.input)

    # Run shape inference so intermediate tensors have shape info
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        print(f"[WARN] Shape inference failed: {e}. Continuing without shapes.")

    reg_outputs, cls_outputs = find_detect_head_convs(model.graph)

    print(f"[INFO] Found 3 regression conv outputs:")
    for name in reg_outputs:
        shape = get_tensor_shape(model.graph, name)
        print(f"  {name} {shape or '(shape unknown)'}")
    print(f"[INFO] Found 3 classification conv outputs:")
    for name in cls_outputs:
        shape = get_tensor_shape(model.graph, name)
        print(f"  {name} {shape or '(shape unknown)'}")

    all_outputs = reg_outputs + cls_outputs
    trimmed = trim_to_outputs(model, all_outputs)

    orig_count = len(model.graph.node)
    trim_count = len(trimmed.graph.node)
    print(f"[INFO] Pruned {orig_count - trim_count} nodes ({orig_count} -> {trim_count})")

    onnx.save(trimmed, args.output)
    print(f"[OK] Trimmed model saved to {args.output}")


if __name__ == "__main__":
    main()
