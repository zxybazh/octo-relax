"""Minimum reproducible example for onnxruntime expand op bug."""
# debugging branch: https://github.com/zxybazh/octo-relax/tree/debug/2022-02-14/bcast_to_legalize

from typing import TYPE_CHECKING, Tuple
import numpy as np
import tvm
from tvm import relax

import onnx
from onnx import helper


from tvm.script import ir as I
from tvm.script import tir as T
from tvm.script import relax as R

metadata = tvm.ir.load_json(
    {
        "root": 1,
        "nodes": [
            {"type_key": ""},
            {"type_key": "Map", "keys": ["relax.expr.Constant"], "data": [2]},
            {"type_key": "Array", "data": [3]},
            {
                "type_key": "relax.expr.Constant",
                "attrs": {"_checked_type_": "10", "data": "0", "span": "0", "struct_info_": "4"},
            },
            {
                "type_key": "relax.TensorStructInfo",
                "attrs": {"dtype": "int64", "ndim": "1", "shape": "5", "span": "0"},
            },
            {
                "type_key": "relax.expr.ShapeExpr",
                "attrs": {"_checked_type_": "9", "span": "0", "struct_info_": "8", "values": "6"},
            },
            {"type_key": "Array", "data": [7]},
            {"type_key": "IntImm", "attrs": {"dtype": "int64", "span": "0", "value": "2"}},
            {
                "type_key": "relax.ShapeStructInfo",
                "attrs": {"ndim": "1", "span": "0", "values": "6"},
            },
            {"type_key": "relax.ShapeType", "attrs": {"ndim": "1", "span": "0"}},
            {
                "type_key": "relax.DynTensorType",
                "attrs": {"dtype": "int64", "ndim": "1", "span": "0"},
            },
        ],
        "b64ndarrays": [
            "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAABAAQACAAAAAAAAABAAAAAAAAAAAwAAAAAAAAAEAAAAAAAAAA=="
        ],
        "attrs": {"tvm_version": "0.11.dev0"},
    }
)


## Dynamic Version
@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((3, 1), dtype="float32")) -> R.Tensor(dtype="float32", ndim=2):
        x_0 = T.Var("x_0", "int64")
        x_1 = T.Var("x_1", "int64")
        with R.dataflow():
            lv: R.Shape(ndim=2) = R.call_packed(
                "vm.builtin.tensor_to_shape",
                metadata["relax.expr.Constant"][0],
                sinfo_args=(R.Shape(ndim=2),),
            )
            lv1: R.Shape([x_0, x_1]) = R.match_cast(lv, R.Shape([x_0, x_1]))
            gv = R.call_tir(broadcast_to, (data,), out_sinfo=R.Tensor((x_0, x_1), dtype="float32"))
            R.output(gv)
        return gv

    @T.prim_func
    def broadcast_to(
        rxplaceholder: T.Buffer((T.int64(3), T.int64(1)), "float32"), var_T_broadcast_to: T.handle
    ):
        T.func_attr({"tir.noalias": True})
        x_0 = T.var("int64")
        x_1 = T.var("int64")
        T_broadcast_to = T.match_buffer(var_T_broadcast_to, (x_0, x_1))
        # with T.block("root"):
        for ax0, ax1 in T.grid(x_0, x_1):
            with T.block("T_broadcast_to"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(rxplaceholder[v_ax0, T.int64(0)])
                T.writes(T_broadcast_to[v_ax0, v_ax1])
                T_broadcast_to[v_ax0, v_ax1] = rxplaceholder[v_ax0, T.int64(0)]


# ## Static Version
# @I.ir_module
# class Module:
#     @R.function
#     def main(data: R.Tensor((3, 1), dtype="float32")) -> R.Tensor(dtype="float32", ndim=2):
#         # x_0 = T.Var("x_0", "int64")
#         # x_1 = T.Var("x_1", "int64")
#         with R.dataflow():
#             lv: R.Shape(ndim=2) = R.call_packed(
#                 "vm.builtin.tensor_to_shape",
#                 metadata["relax.expr.Constant"][0],
#                 sinfo_args=(R.Shape(ndim=2),),
#             )
#             lv1: R.Shape([3, 4]) = R.match_cast(lv, R.Shape([3, 4]))
#             gv = R.call_tir(broadcast_to, (data,), out_sinfo=R.Tensor((3, 4), dtype="float32"))
#             R.output(gv)
#         return gv

#     @T.prim_func
#     def broadcast_to(
#         rxplaceholder: T.Buffer((T.int64(3), T.int64(1)), "float32"),
#         T_broadcast_to: T.Buffer((T.int64(3), T.int64(4)), "float32"),
#     ):
#         T.func_attr({"tir.noalias": True})
#         # x_0 = T.var("int64")
#         # x_1 = T.var("int64")
#         # T_broadcast_to = T.match_buffer(var_T_broadcast_to, (x_0, x_1))
#         # with T.block("root"):
#         for ax0, ax1 in T.grid(3, 4):
#             with T.block("T_broadcast_to"):
#                 v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
#                 T.reads(rxplaceholder[v_ax0, T.int64(0)])
#                 T.writes(T_broadcast_to[v_ax0, v_ax1])
#                 T_broadcast_to[v_ax0, v_ax1] = rxplaceholder[v_ax0, T.int64(0)]


if not TYPE_CHECKING:
    from onnx import TensorProto
else:

    class TensorProto:
        FLOAT = 1
        INT64 = 7


def make_onnx(data: np.ndarray, shape: Tuple[int]):
    """Make onnx model."""
    in_shape = list(data.shape)
    shape_array = np.array(shape)
    shape_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["shape"],
        value=onnx.helper.make_tensor(
            name="const_tensor",
            data_type=TensorProto.INT64,
            dims=shape_array.shape,
            vals=shape_array.flatten().astype("int64"),
        ),
    )
    expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

    in_shape = list(data.shape)
    out_shape = list(shape)

    graph = helper.make_graph(
        [shape_node, expand_node],
        "expand_test",
        inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, in_shape)],
        outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
    )

    model = helper.make_model(graph)
    return model


def main():
    """Main function."""
    in_shape = (3, 1)
    shape = (3, 4)
    data = np.random.uniform(size=in_shape).astype(np.float32)
    onnx_model = make_onnx(data, shape)

    tvm_model = relax.from_onnx(onnx_model, opset=13)
    inputs = {"data": tvm.nd.array(data)}

    # well formed check.
    assert relax.analysis.well_formed(
        tvm_model, check_struct_info=True
    ), "Relax model is not well formed before legalize"

    # # Legalize the relax graph.
    # with tvm.transform.PassContext(opt_level=3):
    #     tvm_model = relax.transform.LegalizeOps()(tvm_model)  # pylint: disable=not-callable
    tvm_model = Module

    # well formed check again.
    assert relax.analysis.well_formed(
        tvm_model, check_struct_info=True
    ), "Relax model is not well formed after legalize"

    # Compile the relax graph into a VM then run.
    target = tvm.target.Target("nvidia/geforce-rtx-3070")
    rt = tvm.cuda()

    func = tvm_model["broadcast_to"]
    mod = tvm.IRModule({"main": func.with_attr("global_symbol", "main")})
    sch = tvm.tir.Schedule(mod)
    loop = sch.fuse(*sch.get_loops("T_broadcast_to"))
    max_threadblocks = 1
    max_threads_per_block = 1
    splits = sch.split(loop, factors=[None, max_threadblocks, max_threads_per_block])
    sch.reorder(splits[1], splits[2], splits[0])

    sch.bind(splits[1], "blockIdx.x")
    sch.bind(splits[2], "threadIdx.x")

    tvm_model["broadcast_to"] = sch.mod["main"]
    tvm_model.show()

    # with tvm.transform.PassContext(opt_level=3):
    #     passes = []
    #     passes.append(relax.transform.RewriteDataflowReshape())
    #     passes.append(relax.transform.ToNonDataflow())
    #     passes.append(relax.transform.CallTIRRewrite())
    #     passes.append(relax.transform.StaticPlanBlockMemory())
    #     passes.append(relax.transform.VMBuiltinLower())
    #     passes.append(relax.transform.VMShapeLower())
    #     # passes.append(relax.transform.AttachGlobalSymbol())
    #     seq = tvm.transform.Sequential(passes)
    #     tvm_model = seq(tvm_model)

    from tvm import meta_schedule as ms
    import tempfile

    # shape_func = tvm_model["shape_func"]
    # mod = tvm.IRModule({"main": shape_func.with_attr("global_symbol", "main")})
    # sch = tvm.tir.Schedule(mod)

    # with tempfile.TemporaryDirectory() as work_dir:
    #     db = ms.tune_tir(mod, target=target, work_dir=work_dir, max_trials_global=4)
    #     (record,) = db.get_top_k(db.commit_workload(mod), 1)
    #     record.trace.apply_to_schedule(sch)

    # breakpoint()
    # from tvm.relax.transform.tuning_api import Trace
    # import tempfile

    # with tempfile.TemporaryDirectory() as work_dir:
    #     with tvm.target.Target(target), tvm.transform.PassContext(
    #         opt_level=3, trace=Trace(tvm_model)
    #     ):
    #         tvm_model = relax.transform.LegalizeOps()(tvm_model)
    #         tuning_pass = relax.transform.MetaScheduleTuneTIR(
    #             work_dir=work_dir, max_trials_global=4
    #         )
    #         tvm_model = tuning_pass(tvm_model)
    #         application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir=work_dir)
    #         tvm_model = application_pass(tvm_model)

    with tvm.transform.PassContext(opt_level=3):
        ex = relax.vm.build(tvm_model, target=target)
        vm = relax.VirtualMachine(ex, rt)

    vm.set_input("main", **inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    # Wrap as a list if there is only one output.
    if isinstance(tvm_output, tvm.nd.NDArray):
        tvm_output = [tvm_output]

    print(data)
    print(tvm_output)


if __name__ == "__main__":
    main()
