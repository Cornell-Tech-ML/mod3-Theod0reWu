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
(324)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, C:\Users\wut6\CT_Courses\classes\MLE\modules\module3\mod3-Theod0reWu\minitorch\fast_ops.py (324)
-------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                               |
    out: Storage,                                                                          |
    out_shape: Shape,                                                                      |
    out_strides: Strides,                                                                  |
    a_storage: Storage,                                                                    |
    a_shape: Shape,                                                                        |
    a_strides: Strides,                                                                    |
    b_storage: Storage,                                                                    |
    b_shape: Shape,                                                                        |
    b_strides: Strides,                                                                    |
) -> None:                                                                                 |
    """NUMBA tensor matrix multiply function.                                              |
                                                                                           |
    Should work for any tensor shapes that broadcast as long as                            |
                                                                                           |
    ```                                                                                    |
    assert a_shape[-1] == b_shape[-2]                                                      |
    ```                                                                                    |
                                                                                           |
    Optimizations:                                                                         |
                                                                                           |
    * Outer loop in parallel                                                               |
    * No index buffers or function calls                                                   |
    * Inner loop should have no global writes, 1 multiply.                                 |
                                                                                           |
                                                                                           |
    Args:                                                                                  |
    ----                                                                                   |
        out (Storage): storage for `out` tensor                                            |
        out_shape (Shape): shape for `out` tensor                                          |
        out_strides (Strides): strides for `out` tensor                                    |
        a_storage (Storage): storage for `a` tensor                                        |
        a_shape (Shape): shape for `a` tensor                                              |
        a_strides (Strides): strides for `a` tensor                                        |
        b_storage (Storage): storage for `b` tensor                                        |
        b_shape (Shape): shape for `b` tensor                                              |
        b_strides (Strides): strides for `b` tensor                                        |
                                                                                           |
    Returns:                                                                               |
    -------                                                                                |
        None : Fills in `out`                                                              |
                                                                                           |
    """                                                                                    |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                 |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                 |
                                                                                           |
    for flat_out_idx in prange(len(out)):--------------------------------------------------| #14
        out_batch = (flat_out_idx // out_shape[-1]) // out_shape[-2] % out_shape[0]        |
        row = (flat_out_idx // out_shape[-1]) % out_shape[-2]                              |
        col = flat_out_idx % out_shape[-1]                                                 |
                                                                                           |
        a_batch = min(a_shape[0] - 1, out_batch)                                           |
        b_batch = min(b_shape[0] - 1, out_batch)                                           |
                                                                                           |
        temp = 0                                                                           |
        for i in prange(a_shape[-1]):------------------------------------------------------| #13
            a_flat_idx = (                                                                 |
                a_batch_stride * a_batch + a_strides[-2] * row + a_strides[-1] * i         |
            )                                                                              |
            b_flat_idx = (                                                                 |
                b_batch_stride * b_batch + b_strides[-2] * i + b_strides[-1] * col         |
            )                                                                              |
            temp += a_storage[a_flat_idx] * b_storage[b_flat_idx]                          |
        out[flat_out_idx] = temp                                                           |
                                                                                           |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #14, #13).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--14 is a parallel loop
   +--13 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--14 (parallel)
   +--13 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--14 (parallel)
   +--13 (serial)



Parallel region 0 (loop #14) had 0 loop(s) fused and 1 loop(s) serialized as
part of the larger parallel loop (#14).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
```

## Task 3.5
### CPU:
#### Simple Dataset:
![image](https://github.com/user-attachments/assets/286f4cec-b6d3-4933-866b-bb18bf9f87b6) <br>
```
python project\run_fast_tensor.py --BACKEND cpu --HIDDEN 10 --DATASET simple --RATE 0.05
```
```
Epoch  0  loss  6.694203635573675 correct 33            | seconds/epoch 4.460672521591187
Epoch  10  loss  6.604209774000792 correct 33           | seconds/epoch 0.0820732831954956
Epoch  20  loss  5.435242991729091 correct 33           | seconds/epoch 0.0884631872177124
Epoch  30  loss  5.296007093267006 correct 33           | seconds/epoch 0.08755621910095215
Epoch  40  loss  4.361664928070718 correct 33           | seconds/epoch 0.09076514244079589
Epoch  50  loss  4.31557633938437 correct 34            | seconds/epoch 0.08889124393463135
Epoch  60  loss  5.414017494177441 correct 45           | seconds/epoch 0.08355543613433838
Epoch  70  loss  3.1043255938282783 correct 47          | seconds/epoch 0.09111926555633545
Epoch  80  loss  2.313356880331218 correct 47           | seconds/epoch 0.0846409797668457
Epoch  90  loss  2.939466563831282 correct 47           | seconds/epoch 0.08510160446166992
Epoch  100  loss  1.2491732943416947 correct 48         | seconds/epoch 0.08915421962738038
Epoch  110  loss  0.7412104530119965 correct 48         | seconds/epoch 0.08654136657714843
Epoch  120  loss  1.3041463004368485 correct 48         | seconds/epoch 0.08522722721099854
Epoch  130  loss  1.3457031500567567 correct 48         | seconds/epoch 0.08836297988891602
Epoch  140  loss  0.6157825661010372 correct 48         | seconds/epoch 0.09195871353149414
Epoch  150  loss  1.2785619120501546 correct 49         | seconds/epoch 0.08353688716888427
Epoch  160  loss  2.1715982541994663 correct 50         | seconds/epoch 0.08224453926086425
Epoch  170  loss  1.0863523466197582 correct 49         | seconds/epoch 0.08273372650146485
Epoch  180  loss  0.5674147020912746 correct 49         | seconds/epoch 0.08297088146209716
Epoch  190  loss  0.2391916452226811 correct 49         | seconds/epoch 0.09390265941619873
Epoch  200  loss  0.2977859698832533 correct 49         | seconds/epoch 0.08385961055755616
Epoch  210  loss  1.778462783206416 correct 48          | seconds/epoch 0.09438588619232177
Epoch  220  loss  0.6057212653377103 correct 49         | seconds/epoch 0.08237102031707763
Epoch  230  loss  0.13174738552174964 correct 50        | seconds/epoch 0.08827166557312012
Epoch  240  loss  0.9920992766652361 correct 49         | seconds/epoch 0.08444123268127442
Average seconds per epoch: 0.2647964382171631
```

#### Xor Dataset:
![image](https://github.com/user-attachments/assets/b88daf22-6151-460c-b388-517ffa97a8ba) <br>
```
python project\run_fast_tensor.py --BACKEND cpu --HIDDEN 10 --DATASET xor --RATE 0.05
```
```
Epoch  0  loss  6.9499320412610155 correct 27           | seconds/epoch 4.232690691947937
Epoch  10  loss  6.680718351485258 correct 29           | seconds/epoch 0.08263862133026123
Epoch  20  loss  6.478257649596769 correct 29           | seconds/epoch 0.1321509599685669
Epoch  30  loss  6.235700997134086 correct 29           | seconds/epoch 0.20751688480377198
Epoch  40  loss  6.696162180642592 correct 29           | seconds/epoch 0.1339592456817627
Epoch  50  loss  6.457783981819828 correct 29           | seconds/epoch 0.11800076961517333
Epoch  60  loss  6.8513905419056 correct 29             | seconds/epoch 0.11050477027893066
Epoch  70  loss  6.485099911866063 correct 29           | seconds/epoch 0.07498133182525635
Epoch  80  loss  6.6731211140139255 correct 30          | seconds/epoch 0.0840139389038086
Epoch  90  loss  6.247267191187696 correct 31           | seconds/epoch 0.08141176700592041
Epoch  100  loss  6.571867580829746 correct 33          | seconds/epoch 0.07408530712127685
Epoch  110  loss  6.48632979496333 correct 33           | seconds/epoch 0.07485306262969971
Epoch  120  loss  5.802213915831637 correct 33          | seconds/epoch 0.07456557750701905
Epoch  130  loss  6.322363131342205 correct 33          | seconds/epoch 0.1241905689239502
Epoch  140  loss  6.5888674225789465 correct 34         | seconds/epoch 0.08182055950164795
Epoch  150  loss  5.67679676574263 correct 35           | seconds/epoch 0.07236161231994628
Epoch  160  loss  5.076234682226752 correct 40          | seconds/epoch 0.07275407314300537
Epoch  170  loss  5.5137453094881534 correct 40         | seconds/epoch 0.08791151046752929
Epoch  180  loss  4.873144324771402 correct 40          | seconds/epoch 0.07876346111297608
Epoch  190  loss  4.045246018321409 correct 43          | seconds/epoch 0.10508003234863281
Epoch  200  loss  4.703722452793134 correct 45          | seconds/epoch 0.11387906074523926
Epoch  210  loss  4.182384389514874 correct 48          | seconds/epoch 0.188634991645813
Epoch  220  loss  4.295865900786416 correct 48          | seconds/epoch 0.12263360023498535
Epoch  230  loss  3.1700340505139692 correct 47         | seconds/epoch 0.10117194652557374
Epoch  240  loss  5.93811275312501 correct 48           | seconds/epoch 0.07759521007537842
Average seconds per epoch: 0.27109865856170656
```

#### Split Dataset:
![image](https://github.com/user-attachments/assets/467cf62e-14ab-4149-8aac-a1e79dff2e8b) <br>
```
python project\run_fast_tensor.py --BACKEND cpu --HIDDEN 10 --DATASET split --RATE 0.05
```
```
Epoch  0  loss  6.935902595900606 correct 31            | seconds/epoch 4.5338050365448
Epoch  10  loss  6.5345083996016955 correct 31          | seconds/epoch 0.06982169151306153
Epoch  20  loss  7.1491917145787225 correct 31          | seconds/epoch 0.1089308500289917
Epoch  30  loss  7.553448914887395 correct 31           | seconds/epoch 0.08967268466949463
Epoch  40  loss  5.383259197693861 correct 31           | seconds/epoch 0.1027400016784668
Epoch  50  loss  7.335481687266494 correct 33           | seconds/epoch 0.15516252517700196
Epoch  60  loss  6.737787498860004 correct 34           | seconds/epoch 0.16844379901885986
Epoch  70  loss  5.40033459288589 correct 35            | seconds/epoch 0.1969202995300293
Epoch  80  loss  5.263538726585789 correct 35           | seconds/epoch 0.2520568609237671
Epoch  90  loss  5.436105995819786 correct 38           | seconds/epoch 0.11309430599212647
Epoch  100  loss  5.647079676491958 correct 39          | seconds/epoch 0.10670986175537109
Epoch  110  loss  5.542330540360641 correct 39          | seconds/epoch 0.10156495571136474
Epoch  120  loss  5.103749143151281 correct 39          | seconds/epoch 0.08840079307556152
Epoch  130  loss  5.083893260347859 correct 39          | seconds/epoch 0.08705761432647705
Epoch  140  loss  5.179255654596995 correct 40          | seconds/epoch 0.08842010498046875
Epoch  150  loss  4.4809166593626895 correct 40         | seconds/epoch 0.08681063652038574
Epoch  160  loss  4.474196452216365 correct 40          | seconds/epoch 0.09185161590576171
Epoch  170  loss  3.484530922709659 correct 42          | seconds/epoch 0.09479870796203613
Epoch  180  loss  3.0261331089061283 correct 43         | seconds/epoch 0.08467631340026856
Epoch  190  loss  4.063270577053808 correct 47          | seconds/epoch 0.09432485103607177
Epoch  200  loss  3.0921109705656216 correct 47         | seconds/epoch 0.07866024971008301
Epoch  210  loss  1.9668775297020231 correct 47         | seconds/epoch 0.08236241340637207
Epoch  220  loss  2.450912121268675 correct 46          | seconds/epoch 0.07480595111846924
Epoch  230  loss  1.7128356194888033 correct 46         | seconds/epoch 0.08092434406280517
Epoch  240  loss  2.2042711365071854 correct 48         | seconds/epoch 0.07724161148071289
Average seconds per epoch: 0.2872714443206787
```

### GPU (Run on my laptop):
#### Simple Dataset:
Command used:
```
python project\run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --RATE 0.05
```
Logs (avg seconds per epochs included at the end):
```
Epoch  0  loss  5.712280438235494 correct 41            | seconds/epoch 0.4818134784698486
Epoch  10  loss  2.867498083201923 correct 48           | seconds/epoch 2.143973731994629
Epoch  20  loss  2.018798982441157 correct 49           | seconds/epoch 2.207001781463623
Epoch  30  loss  1.4005269425124423 correct 50          | seconds/epoch 2.315777134895325
Epoch  40  loss  1.4676480775117842 correct 49          | seconds/epoch 2.429712152481079
Epoch  50  loss  1.2504447867405089 correct 50          | seconds/epoch 2.523483467102051
Epoch  60  loss  1.6849641662934356 correct 48          | seconds/epoch 2.5539917230606077
Epoch  70  loss  0.6382444030721857 correct 45          | seconds/epoch 2.4969515800476074
Epoch  80  loss  0.600170589576513 correct 49           | seconds/epoch 2.551994466781616
Epoch  90  loss  0.930862726471215 correct 49           | seconds/epoch 2.42630398273468
Epoch  100  loss  0.6066266035347474 correct 49         | seconds/epoch 2.481922149658203
Epoch  110  loss  1.4625812031912568 correct 50         | seconds/epoch 2.5127543210983276
Epoch  120  loss  0.8303106696265536 correct 50         | seconds/epoch 2.481510615348816
Epoch  130  loss  0.42343751917821876 correct 48        | seconds/epoch 3.0763891458511354
Epoch  140  loss  0.6088427688131424 correct 50         | seconds/epoch 2.4379791259765624
Epoch  150  loss  1.1494899270340662 correct 49         | seconds/epoch 2.335973310470581
Epoch  160  loss  0.3104813316681823 correct 47         | seconds/epoch 2.3209951877593995
Epoch  170  loss  0.6490545865281871 correct 50         | seconds/epoch 2.4180409908294678
Epoch  180  loss  0.007814803652277866 correct 50       | seconds/epoch 2.4579798936843873
Epoch  190  loss  0.2479955071098627 correct 49         | seconds/epoch 2.441990518569946
Epoch  200  loss  0.36202959333035006 correct 50        | seconds/epoch 2.499089741706848
Epoch  210  loss  0.1764431456956813 correct 49         | seconds/epoch 2.548389935493469
Epoch  220  loss  0.8230358457622731 correct 50         | seconds/epoch 2.4629998207092285
Epoch  230  loss  0.6290699766775337 correct 50         | seconds/epoch 2.4849561929702757
Epoch  240  loss  0.4568363700288108 correct 50         | seconds/epoch 2.5850056171417237
Average seconds per epoch: 2.4880107002258303
```
#### Split Dataset:
Command used:
```
project\run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05 <br>
```
Logs (avg seconds per epochs included at the end):
```
Epoch  0  loss  9.143952981386708 correct 27            | seconds/epoch 0.4588161468505859
Epoch  10  loss  6.647929532827963 correct 43           | seconds/epoch 2.382473683357239
Epoch  20  loss  2.989496865015149 correct 46           | seconds/epoch 2.342395544052124
Epoch  30  loss  4.160345482993819 correct 45           | seconds/epoch 2.4269953966140747
Epoch  40  loss  1.8624833496298276 correct 47          | seconds/epoch 2.592983102798462
Epoch  50  loss  4.90103776350167 correct 47            | seconds/epoch 2.533026361465454
Epoch  60  loss  0.9670231205602602 correct 48          | seconds/epoch 2.6259927988052367
Epoch  70  loss  2.509878031695367 correct 48           | seconds/epoch 2.5039510250091555
Epoch  80  loss  1.541006965697605 correct 48           | seconds/epoch 2.5460246801376343
Epoch  90  loss  1.9394090580508092 correct 48          | seconds/epoch 5.464270830154419
Epoch  100  loss  0.6990160587323384 correct 48         | seconds/epoch 5.758680987358093
Epoch  110  loss  0.9797124534170019 correct 48         | seconds/epoch 3.155682897567749
Epoch  120  loss  0.9001095478523986 correct 46         | seconds/epoch 2.4738502979278563
Epoch  130  loss  1.0872823600962285 correct 49         | seconds/epoch 2.3628488779067993
Epoch  140  loss  1.1730057645140373 correct 50         | seconds/epoch 2.473087787628174
Epoch  150  loss  1.4212014227918246 correct 50         | seconds/epoch 2.619302272796631
Epoch  160  loss  0.3285771990942866 correct 50         | seconds/epoch 2.7005623817443847
Epoch  170  loss  0.38851232612125264 correct 50        | seconds/epoch 2.620073366165161
Epoch  180  loss  0.7348713957235404 correct 49         | seconds/epoch 2.7077195405960084
Epoch  190  loss  0.47030230672319095 correct 50        | seconds/epoch 2.6228098630905152
Epoch  200  loss  0.3311886733788445 correct 50         | seconds/epoch 3.1659783124923706
Epoch  210  loss  0.7094557045010786 correct 50         | seconds/epoch 2.698988509178162
Epoch  220  loss  0.5464181139030917 correct 49         | seconds/epoch 3.0223538637161256
Epoch  230  loss  1.0552813680798356 correct 50         | seconds/epoch 2.669624614715576
Epoch  240  loss  0.7332071265043433 correct 50         | seconds/epoch 2.579005002975464
Average seconds per epoch: 2.8711271839141848
```
#### Xor Dataset:
Command used:
```
python project\run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.03 <br>
```
Logs (avg seconds per epochs included at the end):
```
Epoch  0  loss  8.412865411870944 correct 27            | seconds/epoch 0.4975917339324951
Epoch  10  loss  7.9244930991169 correct 38             | seconds/epoch 2.3988871574401855
Epoch  20  loss  4.5187382825404905 correct 44          | seconds/epoch 2.631600022315979
Epoch  30  loss  4.475804539940274 correct 42           | seconds/epoch 2.54698965549469
Epoch  40  loss  3.204916566737308 correct 42           | seconds/epoch 2.6560134410858156
Epoch  50  loss  4.722660304586044 correct 45           | seconds/epoch 2.577858829498291
Epoch  60  loss  3.1400135059628878 correct 41          | seconds/epoch 2.559103488922119
Epoch  70  loss  3.820567399873803 correct 45           | seconds/epoch 2.613305139541626
Epoch  80  loss  3.1506447432400804 correct 47          | seconds/epoch 2.55369291305542
Epoch  90  loss  3.7994214100729544 correct 46          | seconds/epoch 2.608956217765808
Epoch  100  loss  2.185394644091533 correct 47          | seconds/epoch 2.592012095451355
Epoch  110  loss  2.3536728020889996 correct 47         | seconds/epoch 2.554976201057434
Epoch  120  loss  2.355381803530649 correct 48          | seconds/epoch 2.602019739151001
Epoch  130  loss  3.5170862203287214 correct 47         | seconds/epoch 2.5819918870925904
Epoch  140  loss  1.439168807182527 correct 49          | seconds/epoch 2.5659679651260374
Epoch  150  loss  0.9100422137634152 correct 50         | seconds/epoch 2.940946936607361
Epoch  160  loss  1.8288662836382121 correct 46         | seconds/epoch 2.2495548248291017
Epoch  170  loss  2.0160235126716826 correct 48         | seconds/epoch 2.279972958564758
Epoch  180  loss  3.167623722978433 correct 46          | seconds/epoch 2.3555174112319945
Epoch  190  loss  1.8885906416265215 correct 49         | seconds/epoch 2.5094778537750244
Epoch  200  loss  1.8082588528691947 correct 50         | seconds/epoch 2.5830110549926757
Epoch  210  loss  0.8974956248110183 correct 50         | seconds/epoch 2.4644993782043456
Epoch  220  loss  1.922055767402042 correct 50          | seconds/epoch 2.484434151649475
Epoch  230  loss  0.8809161974312701 correct 50         | seconds/epoch 2.5250071048736573
Epoch  240  loss  1.6102697098171537 correct 50         | seconds/epoch 4.155127501487732
Average seconds per epoch: 2.6163352003097535
```
### BIG model (Run on my laptop)
#### GPU
Command used:
```
python project\run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET circle --RATE 0.05
```
Logs (avg seconds per epochs included at the end):
```
Epoch  0  loss  6.870596813366973 correct 30            | seconds/epoch 0.5430021762847901
Epoch  10  loss  4.0845138142409905 correct 44          | seconds/epoch 2.71392936706543
Epoch  20  loss  3.339567267838176 correct 47           | seconds/epoch 2.828413224220276
Epoch  30  loss  2.2385746780700755 correct 49          | seconds/epoch 2.589972400665283
Epoch  40  loss  1.9128675856754316 correct 47          | seconds/epoch 2.757995915412903
Epoch  50  loss  2.436322016928945 correct 50           | seconds/epoch 2.789018726348877
Epoch  60  loss  1.2177436811565368 correct 49          | seconds/epoch 3.010077786445618
Epoch  70  loss  1.3822464287138623 correct 50          | seconds/epoch 2.702754592895508
Epoch  80  loss  1.3156974456419632 correct 50          | seconds/epoch 2.834112620353699
Epoch  90  loss  3.581515339179111 correct 45           | seconds/epoch 2.6999956130981446
Epoch  100  loss  0.7966047694622819 correct 50         | seconds/epoch 2.6626018285751343
Epoch  110  loss  0.35705334150482393 correct 48        | seconds/epoch 2.7353808164596556
Epoch  120  loss  1.7750754376400435 correct 49         | seconds/epoch 2.6449970483779905
Epoch  130  loss  0.9054355660122215 correct 50         | seconds/epoch 2.6685071706771852
Epoch  140  loss  0.23733862325040667 correct 50        | seconds/epoch 2.7434858322143554
Epoch  150  loss  0.668605276504419 correct 50          | seconds/epoch 2.650985908508301
Epoch  160  loss  1.482401228858011 correct 50          | seconds/epoch 2.916987180709839
Epoch  170  loss  1.013457983606611 correct 50          | seconds/epoch 2.6619905710220335
Epoch  180  loss  1.2157497002516897 correct 50         | seconds/epoch 2.6795185804367065
Epoch  190  loss  0.7009944966020022 correct 49         | seconds/epoch 2.6424423694610595
Epoch  200  loss  0.23236207162135963 correct 50        | seconds/epoch 2.6676048040390015
Epoch  210  loss  0.9008350983278568 correct 49         | seconds/epoch 2.6104205369949343
Epoch  220  loss  0.10584503177511204 correct 50        | seconds/epoch 2.6669480562210084
Epoch  230  loss  0.35961149897931344 correct 50        | seconds/epoch 2.635998797416687
Epoch  240  loss  0.372198157242294 correct 50          | seconds/epoch 2.69099907875061
Average seconds per epoch: 2.7266457195281983
```
Command used:
```
python project\run_fast_tensor.py --BACKEND gpu --HIDDEN 300 --DATASET circle --RATE 0.05
```
Logs (avg seconds per epochs included at the end):
```
Epoch  0  loss  28.171468382642345 correct 31           | seconds/epoch 0.5043039321899414
Epoch  10  loss  5.504379392778562 correct 34           | seconds/epoch 2.5479716062545776
Epoch  20  loss  2.892777896558108 correct 45           | seconds/epoch 2.6510010957717896
Epoch  30  loss  1.1784161224573164 correct 49          | seconds/epoch 2.7401905059814453
Epoch  40  loss  2.100523219333091 correct 47           | seconds/epoch 2.9033321142196655
Epoch  50  loss  0.8396566449045946 correct 49          | seconds/epoch 3.0194505214691163
Epoch  60  loss  0.8505211808182975 correct 50          | seconds/epoch 3.016508913040161
Epoch  70  loss  1.198139901654135 correct 49           | seconds/epoch 3.14146032333374
Epoch  80  loss  0.7681456193788955 correct 50          | seconds/epoch 3.089622211456299
Epoch  90  loss  1.659390149746025 correct 46           | seconds/epoch 3.0787816047668457
Epoch  100  loss  0.6064169360020256 correct 50         | seconds/epoch 3.070576000213623
Epoch  110  loss  0.6560253968551057 correct 50         | seconds/epoch 3.063985991477966
Epoch  120  loss  0.8209059122144324 correct 50         | seconds/epoch 3.230097699165344
Epoch  130  loss  0.4504267705138605 correct 50         | seconds/epoch 2.6828840732574464
Epoch  140  loss  0.09870575497169853 correct 50        | seconds/epoch 2.7489930152893067
Epoch  150  loss  1.45460689743168 correct 46           | seconds/epoch 2.9020030975341795
Epoch  160  loss  0.9734346643135217 correct 49         | seconds/epoch 3.546258974075317
Epoch  170  loss  0.7376466003399689 correct 50         | seconds/epoch 3.8930065631866455
Epoch  180  loss  0.2509479117058339 correct 50         | seconds/epoch 4.131708931922913
Epoch  190  loss  0.4320897970405287 correct 50         | seconds/epoch 3.344966745376587
Epoch  200  loss  0.1413876955933165 correct 50         | seconds/epoch 3.366779112815857
Epoch  210  loss  0.35963824877333406 correct 50        | seconds/epoch 3.153874325752258
Epoch  220  loss  0.6084173162664974 correct 50         | seconds/epoch 3.1015902280807497
Epoch  230  loss  0.6714454235433175 correct 50         | seconds/epoch 3.127086901664734
Epoch  240  loss  0.47823485237761115 correct 50        | seconds/epoch 3.0469990015029906
Average seconds per epoch: 3.1182627954483033
```

#### CPU
Command used:
```
python project\run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET circle --RATE 0.05
```
Logs (avg seconds per epochs included at the end):
```
Epoch  0  loss  6.104287598709402 correct 28            | seconds/epoch 2.9168707847595217
Epoch  10  loss  1.8608152285782513 correct 35          | seconds/epoch 0.21512224674224853
Epoch  20  loss  3.1055171304495572 correct 36          | seconds/epoch 0.21822071075439453
Epoch  30  loss  2.869994272397738 correct 45           | seconds/epoch 0.22906582355499266
Epoch  40  loss  1.8694434093313785 correct 44          | seconds/epoch 0.28166120052337645
Epoch  50  loss  2.417625082894242 correct 48           | seconds/epoch 0.25356497764587405
Epoch  60  loss  1.620660631838246 correct 48           | seconds/epoch 0.26378912925720216
Epoch  70  loss  2.519918651327616 correct 47           | seconds/epoch 0.25952582359313964
Epoch  80  loss  2.8148612059736067 correct 46          | seconds/epoch 0.24199347496032714
Epoch  90  loss  2.7504388610935924 correct 50          | seconds/epoch 0.2515801668167114
Epoch  100  loss  1.695262024940334 correct 49          | seconds/epoch 0.2637979030609131
Epoch  110  loss  1.0248643459970164 correct 49         | seconds/epoch 0.26399712562561034
Epoch  120  loss  1.7144258782483055 correct 50         | seconds/epoch 0.23165342807769776
Epoch  130  loss  0.262755789481371 correct 48          | seconds/epoch 0.20922579765319824
Epoch  140  loss  1.9364652587842814 correct 49         | seconds/epoch 0.2474158763885498
Epoch  150  loss  2.1622697826036 correct 47            | seconds/epoch 0.25349810123443606
Epoch  160  loss  1.5256301975552111 correct 50         | seconds/epoch 0.2433178424835205
Epoch  170  loss  1.0624079177370778 correct 48         | seconds/epoch 0.2691028118133545
Epoch  180  loss  1.6691962085327525 correct 50         | seconds/epoch 0.23237910270690917
Epoch  190  loss  2.6056218758128753 correct 48         | seconds/epoch 0.21966190338134767
Epoch  200  loss  1.3103482777626654 correct 48         | seconds/epoch 0.22173237800598145
Epoch  210  loss  1.3110849558501982 correct 50         | seconds/epoch 0.22077481746673583
Epoch  220  loss  0.799661138364357 correct 49          | seconds/epoch 0.2225165843963623
Epoch  230  loss  0.33856573971081694 correct 50        | seconds/epoch 0.22598445415496826
Epoch  240  loss  1.351554563579143 correct 50          | seconds/epoch 0.22698285579681396
Average seconds per epoch: 0.35551807975769045
```
Command used:
```
python project\run_fast_tensor.py --BACKEND cpu --HIDDEN 300 --DATASET circle --RATE 0.05
```
Logs (avg seconds per epochs included at the end):
```
Epoch  0  loss  27.631012649684294 correct 38           | seconds/epoch 3.005239748954773
Epoch  10  loss  69.07748748640782 correct 38           | seconds/epoch 0.39199080467224123
Epoch  20  loss  27.631011329358284 correct 38          | seconds/epoch 0.39103991985321046
Epoch  30  loss  55.26198286002704 correct 38           | seconds/epoch 0.40619046688079835
Epoch  40  loss  41.446503275252724 correct 38          | seconds/epoch 0.41067817211151125
Epoch  50  loss  27.63099261561331 correct 38           | seconds/epoch 0.4100408315658569
Epoch  60  loss  27.63096543105489 correct 38           | seconds/epoch 0.4159677982330322
Epoch  70  loss  27.631011366511263 correct 38          | seconds/epoch 0.43390014171600344
Epoch  80  loss  41.44651886763134 correct 38           | seconds/epoch 0.4146789312362671
Epoch  90  loss  13.815481527866377 correct 38          | seconds/epoch 0.40773146152496337
Epoch  100  loss  41.44648865050749 correct 38          | seconds/epoch 0.39917242527008057
Epoch  110  loss  -9.984531534735634e-06 correct 38     | seconds/epoch 0.4026747941970825
Epoch  120  loss  41.44649570136285 correct 38          | seconds/epoch 0.3911685228347778
Epoch  130  loss  13.815480438410303 correct 38         | seconds/epoch 0.3914589166641235
Epoch  140  loss  55.26199444889932 correct 38          | seconds/epoch 0.3999034404754639
Epoch  150  loss  55.262013868096034 correct 38         | seconds/epoch 0.4013087749481201
Epoch  160  loss  27.630991620125997 correct 38         | seconds/epoch 0.3961567640304565
Epoch  170  loss  13.815501189430222 correct 38         | seconds/epoch 0.3889441013336182
Epoch  180  loss  55.261971951663526 correct 38         | seconds/epoch 0.38910181522369386
Epoch  190  loss  41.446495047203236 correct 38         | seconds/epoch 0.39459052085876467
Epoch  200  loss  13.815501550541192 correct 38         | seconds/epoch 0.4124325752258301
Epoch  210  loss  55.26197641211142 correct 38          | seconds/epoch 0.41915814876556395
Epoch  220  loss  13.81547358707588 correct 38          | seconds/epoch 0.4181068420410156
Epoch  230  loss  41.44649621971636 correct 38          | seconds/epoch 0.39754886627197267
Epoch  240  loss  13.81550162851163 correct 38          | seconds/epoch 0.3885569334030151
Average seconds per epoch: 0.5210476131439209
```

### Graph for matrix multiply speedup
![image](https://github.com/user-attachments/assets/170e205d-9ee1-4b77-ac09-068231fcb66f)

