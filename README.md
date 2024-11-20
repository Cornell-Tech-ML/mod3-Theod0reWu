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

## Task 3.1
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

## Task 3.2
The output for matrix multiply:
```
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, C:\Users\wu
t6\CT_Courses\classes\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py
(314)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\wut6\CT_Courses\classes\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (314)
--------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                      |
    out: Storage,                                                                                 |
    out_shape: Shape,                                                                             |
    out_strides: Strides,                                                                         |
    a_storage: Storage,                                                                           |
    a_shape: Shape,                                                                               |
    a_strides: Strides,                                                                           |
    b_storage: Storage,                                                                           |
    b_shape: Shape,                                                                               |
    b_strides: Strides,                                                                           |
) -> None:                                                                                        |
    """NUMBA tensor matrix multiply function.                                                     |
                                                                                                  |
    Should work for any tensor shapes that broadcast as long as                                   |
                                                                                                  |
    ```                                                                                           |
    assert a_shape[-1] == b_shape[-2]                                                             |
    ```                                                                                           |
                                                                                                  |
    Optimizations:                                                                                |
                                                                                                  |
    * Outer loop in parallel                                                                      |
    * No index buffers or function calls                                                          |
    * Inner loop should have no global writes, 1 multiply.                                        |
                                                                                                  |
                                                                                                  |
    Args:                                                                                         |
    ----                                                                                          |
        out (Storage): storage for `out` tensor                                                   |
        out_shape (Shape): shape for `out` tensor                                                 |
        out_strides (Strides): strides for `out` tensor                                           |
        a_storage (Storage): storage for `a` tensor                                               |
        a_shape (Shape): shape for `a` tensor                                                     |
        a_strides (Strides): strides for `a` tensor                                               |
        b_storage (Storage): storage for `b` tensor                                               |
        b_shape (Shape): shape for `b` tensor                                                     |
        b_strides (Strides): strides for `b` tensor                                               |
                                                                                                  |
    Returns:                                                                                      |
    -------                                                                                       |
        None : Fills in `out`                                                                     |
                                                                                                  |
    """                                                                                           |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                        |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                        |
                                                                                                  |
    for flat_out_idx in prange(len(out)):---------------------------------------------------------| #15
        out_index = np.empty(MAX_DIMS, np.int32)                                                  |
        a_index = np.empty(MAX_DIMS, np.int32)                                                    |
        b_index = np.empty(MAX_DIMS, np.int32)                                                    |
                                                                                                  |
        to_index(flat_out_idx, out_shape, out_index)                                              |
        row, col = out_index[len(a_shape) - 2], out_index[len(a_shape) - 1]                       |
        out_index[len(a_shape) - 2:] = 0----------------------------------------------------------| #13
        broadcast_index(out_index, out_shape, a_shape, a_index)                                   |
        broadcast_index(out_index, out_shape, b_shape, b_index)                                   |
                                                                                                  |
        temp = 0                                                                                  |
        for i in prange(a_shape[-1]):-------------------------------------------------------------| #14
            a_flat_idx = a_batch_stride * a_index[0] + a_strides[-2] * row + a_strides[-1] * i    |
            b_flat_idx = b_batch_stride * b_index[0] + b_strides[-2] * i + b_strides[-1] * col    |
            temp += a_storage[a_flat_idx] * b_storage[b_flat_idx]                                 |
        out[flat_out_idx] = temp                                                                  |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #15).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--15 is a parallel loop
   +--13 --> rewritten as a serial loop
   +--14 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--15 (parallel)
   +--13 (parallel)
   +--14 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--15 (parallel)
   +--13 (serial)
   +--14 (serial)



Parallel region 0 (loop #15) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#15).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (361) is
hoisted out of the parallel loop labelled #15 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (362) is
hoisted out of the parallel loop labelled #15 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: a_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at C:\Users\wut6\CT_Courses\c
lasses\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (363) is
hoisted out of the parallel loop labelled #15 (it will be performed before the
loop is executed and reused inside the loop):
   Allocation:: b_index = np.empty(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
```

## Task 3.5
### CPU:
Simple Dataset: <br>
![image](https://github.com/user-attachments/assets/286f4cec-b6d3-4933-866b-bb18bf9f87b6) <br>
Xor Dataset: <br>
![image](https://github.com/user-attachments/assets/b88daf22-6151-460c-b388-517ffa97a8ba) <br>
Split Dataset: <br>
![image](https://github.com/user-attachments/assets/467cf62e-14ab-4149-8aac-a1e79dff2e8b) <br>

### GPU:
Split Dataset: <br>
project\run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```
Epoch  0  loss  5.9280363848075375 correct 31
Epoch  10  loss  6.985445647831394 correct 40
Epoch  20  loss  3.6903102406397545 correct 44
Epoch  30  loss  5.448372899298298 correct 42
Epoch  40  loss  2.7511269989319462 correct 44
Epoch  50  loss  2.6903374972210665 correct 44
Epoch  60  loss  2.8767859013677537 correct 47
Epoch  70  loss  3.365927088997409 correct 47
Epoch  80  loss  1.2426309734667167 correct 45
Epoch  90  loss  2.4999645616895436 correct 48
Epoch  100  loss  3.3021779850483446 correct 49
Epoch  110  loss  1.5878986455651016 correct 48
Epoch  120  loss  1.5846321356594855 correct 48
Epoch  130  loss  2.284663272377156 correct 48
Epoch  140  loss  0.4476190891989245 correct 50
Epoch  150  loss  1.7077667050344094 correct 49
Epoch  160  loss  0.928700402709943 correct 49
Epoch  170  loss  0.5473643660086247 correct 48
Epoch  180  loss  0.9013587663567523 correct 50
Epoch  190  loss  1.1694975442065394 correct 50
Epoch  200  loss  1.4263421041806454 correct 50
Epoch  210  loss  1.853899932078388 correct 50
Epoch  220  loss  0.8293124943184889 correct 50
Epoch  230  loss  0.8190219831228698 correct 50
Epoch  240  loss  0.6978672369666373 correct 49
command took 0:12:40.90 (760.90s total)
```
Seconds per Epoch: 3.0436

### Graph
![image](https://github.com/user-attachments/assets/170e205d-9ee1-4b77-ac09-068231fcb66f)

