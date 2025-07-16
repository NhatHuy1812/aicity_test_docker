#!/usr/bin/env python3
"""
Build a mixed-precision TensorRT engine from ONNX, forcing all normalization layers to FP32
and using an explicit batch optimization profile.
"""
import argparse
import logging
import sys

import tensorrt as trt


def build_engine(onnx_path: str, engine_path: str, workspace_size: int = 4 << 30):
    # Set up Python logging
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("build_trt")

    # Create builder and network (explicit batch)
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse the ONNX model
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            log.error("Failed to parse ONNX model")
            for idx in range(parser.num_errors):
                log.error(parser.get_error(idx))
            sys.exit(1)

    # Configure builder for FP16 with workspace limit
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    config.set_flag(trt.BuilderFlag.FP16)

    # Create optimization profile for dynamic inputs
    profile = builder.create_optimization_profile()
    # images: [batch, 3, 1280, 1280]
    profile.set_shape(
        "images",
        min=(1, 3, 1280, 1280),
        opt=(1, 3, 1280, 1280),
        max=(1, 3, 1280, 1280),
    )
    # orig_target_sizes: [batch, 2]
    profile.set_shape(
        "orig_target_sizes",
        min=(1, 2),
        opt=(1, 2),
        max=(1, 2),
    )
    config.add_optimization_profile(profile)

    # Force all normalization layers (BatchNorm, etc.) to use FP32
    # for i in range(network.num_layers):
    #     layer = network.get_layer(i)
    #     if layer.type == trt.LayerType.NORMALIZATION:
    #         log.info(f"Forcing layer '{layer.name}' to FP32 precision")
    #         layer.precision = trt.DataType.FLOAT
    #         for o in range(layer.num_outputs):
    #             layer.set_output_type(o, trt.DataType.FLOAT)
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        # Check for various normalization layer types
        if layer.type in [trt.LayerType.SCALE, trt.LayerType.ELEMENTWISE]:
            # Additional check for layer names that might indicate normalization
            if any(norm_keyword in layer.name.lower() for norm_keyword in ['norm', 'bn', 'batch_norm', 'layer_norm', 'group_norm']):
                log.info(f"Forcing layer '{layer.name}' to FP32 precision")
                layer.precision = trt.DataType.FLOAT
                for o in range(layer.num_outputs):
                    layer.set_output_type(o, trt.DataType.FLOAT)
        # Also check for any layer with normalization in the name
        elif any(norm_keyword in layer.name.lower() for norm_keyword in ['norm', 'bn', 'batch_norm', 'layer_norm', 'group_norm']):
            log.info(f"Forcing normalization layer '{layer.name}' (type: {layer.type}) to FP32 precision")
            layer.precision = trt.DataType.FLOAT
            for o in range(layer.num_outputs):
                layer.set_output_type(o, trt.DataType.FLOAT)
    # Build the engine
    engine = builder.build_engine(network, config)
    if engine is None:
        log.error("Engine build failed")
        sys.exit(1)

    # Serialize engine to file
    with open(engine_path, "wb") as f:
        log.info(f"Serializing engine to '{engine_path}'")
        f.write(engine.serialize())

    log.info("Engine build complete")


def main():
    parser = argparse.ArgumentParser(
        description="Build mixed-precision TensorRT engine with FP16 convs and FP32 norms"
    )
    parser.add_argument(
        "--onnx", type=str, required=True, help="Path to input ONNX model"
    )
    parser.add_argument(
        "--engine", type=str, required=True, help="Path to output TensorRT engine file"
    )
    parser.add_argument(
        "--workspace", type=int, default=4 << 30,
        help="Workspace size in bytes (default=4GiB)"
    )
    args = parser.parse_args()

    build_engine(args.onnx, args.engine, args.workspace)


if __name__ == "__main__":
    main()
