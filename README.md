# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


The output from parallel_check.py for MAP, ZIP and REDUCE:
```
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, C:\Users\w
ut6\CT_Courses\classes\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py
 (168)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, C:\Users\wut6\CT_Courses\classes\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (168)
-----------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                            |
        out: Storage,                                                                    |
        out_shape: Shape,                                                                |
        out_strides: Strides,                                                            |
        in_storage: Storage,                                                             |
        in_shape: Shape,                                                                 |
        in_strides: Strides,                                                             |
    ) -> None:                                                                           |
        strides_same = check_strides(in_strides, in_shape, out_strides, out_shape)       |
        if (strides_same):                                                               |
            for i in prange(len(out)):---------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                               |
        else:                                                                            |
            for i in prange(len(out)):---------------------------------------------------| #3
                out_index = np.empty(MAX_DIMS, np.int32)                                 |
                in_index = np.empty(MAX_DIMS, np.int32)                                  |
                                                                                         |
                to_index(i, out_shape, out_index)                                        |
                broadcast_index(out_index, out_shape, in_shape, in_index)                |
                j = index_to_position(in_index, in_strides)                              |
                out[i] = fn(in_storage[j])                                               |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (182) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (183) is
hoisted out of the parallel loop labelled #3 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: in_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, C:\Users\w
ut6\CT_Courses\classes\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py
 (216)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, C:\Users\wut6\CT_Courses\classes\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (216)
-----------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                        |
        out: Storage,                                                                                |
        out_shape: Shape,                                                                            |
        out_strides: Strides,                                                                        |
        a_storage: Storage,                                                                          |
        a_shape: Shape,                                                                              |
        a_strides: Strides,                                                                          |
        b_storage: Storage,                                                                          |
        b_shape: Shape,                                                                              |
        b_strides: Strides,                                                                          |
    ) -> None:                                                                                       |
        strides_same_a = check_strides(a_strides, a_shape, out_strides, out_shape)                   |
        strides_same_b = check_strides(b_strides, b_shape, out_strides, out_shape)                   |
        # print("a:", strides_same_a, "| b:", strides_same_b)                                        |
        if (strides_same_a and strides_same_b):                                                      |
            for out_flat_idx in prange(len(out)):----------------------------------------------------| #8
                out[out_flat_idx] = fn(a_storage[out_flat_idx], b_storage[out_flat_idx])             |
        elif (strides_same_a):                                                                       |
            for out_flat_idx in prange(len(out)):----------------------------------------------------| #10
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                                       |
                b_index = np.empty(MAX_DIMS, dtype=np.int32)                                         |
                to_index(out_flat_idx, out_shape, out_index)                                         |
                                                                                                     |
                broadcast_index(out_index, out_shape, b_shape, b_index)                              |
                b_flat_idx = index_to_position(b_index, b_strides)                                   |
                                                                                                     |
                out[out_flat_idx] = fn(a_storage[out_flat_idx], b_storage[b_flat_idx])               |
        elif (strides_same_b):                                                                       |
            for out_flat_idx in prange(len(out)):----------------------------------------------------| #9
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                                       |
                a_index = np.empty(MAX_DIMS, dtype=np.int32)                                         |
                to_index(out_flat_idx, out_shape, out_index)                                         |
                                                                                                     |
                broadcast_index(out_index, out_shape, a_shape, a_index)                              |
                a_flat_idx = index_to_position(a_index, a_strides)                                   |
                                                                                                     |
                out[out_flat_idx] = fn(a_storage[a_flat_idx], b_storage[out_flat_idx])               |
        else:                                                                                        |
            for out_flat_idx in prange(len(out)):----------------------------------------------------| #11
                out_index = np.empty(MAX_DIMS, dtype=np.int32)                                       |
                a_index = np.empty(MAX_DIMS, dtype=np.int32)                                         |
                b_index = np.empty(MAX_DIMS, dtype=np.int32)                                         |
                                                                                                     |
                to_index(out_flat_idx, out_shape, out_index)                                         |
                                                                                                     |
                broadcast_index(out_index, out_shape, a_shape, a_index)                              |
                a_flat_idx = index_to_position(a_index, a_strides)                                   |
                                                                                                     |
                broadcast_index(out_index, out_shape, b_shape, b_index)                              |
                b_flat_idx = index_to_position(b_index, b_strides)                                   |
                                                                                                     |
                out[out_flat_idx] = fn(a_storage[a_flat_idx], b_storage[b_flat_idx])                 |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 8 parallel for-
loop(s) (originating from loops labelled: #8, #10, #9, #11, #4, #5, #6, #7).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (235) is
hoisted out of the parallel loop labelled #10 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (236) is
hoisted out of the parallel loop labelled #10 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (245) is
hoisted out of the parallel loop labelled #9 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (246) is
hoisted out of the parallel loop labelled #9 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (255) is
hoisted out of the parallel loop labelled #11 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (256) is
hoisted out of the parallel loop labelled #11 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (257) is
hoisted out of the parallel loop labelled #11 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, C:\U
sers\wut6\CT_Courses\classes\MLE\modules\module3\mod3-
Theod0reWu\minitorch\fast_ops.py (293)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, C:\Users\wut6\CT_Courses\classes\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (293)
---------------------------------------------------------------|loop #ID
    def _reduce(                                               |
        out: Storage,                                          |
        out_shape: Shape,                                      |
        out_strides: Strides,                                  |
        a_storage: Storage,                                    |
        a_shape: Shape,                                        |
        a_strides: Strides,                                    |
        reduce_dim: int,                                       |
    ) -> None:                                                 |
        reduce_size = a_shape[reduce_dim]                      |
        for i in prange(len(out)):-----------------------------| #12
            out_index = np.empty(MAX_DIMS, np.int32)           |
            to_index(i, out_shape, out_index)                  |
            for s in range(reduce_size):                       |
                out_index[reduce_dim] = s                      |
                j = index_to_position(out_index, a_strides)    |
                out[i] = fn(out[i], a_storage[j])              |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #12).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (304) is
hoisted out of the parallel loop labelled #12 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
```
