# PyTorchPlus

![PyTorch](https://github.com/Phoenix1153/20TeamA/blob/master/picture/LOGO.png?raw=true)

## Summary

**PyTorchPlus** is a compositive package of deep learning algorithms involving areas of *adversarial attack*, *model compression*, *object detection*, *reading comprehension* and *active learning*.

## Dependencies

General dependencies for installing **PyTorchPlus**

| **python**       | **>=3.6**    |
| :--------------- | ------------ |
| **PyTorch**      | **>=1.1.0**  |
| **Numpy**        | **1.18.4**   |
| **Plckle**       | **4.0**      |
| **Pillow**       | **6.0**      |
| **CUDA**         | **10.0**     |
| **scipy**        | **1.3.2**    |
| **scikit-learn** | **0.22.1**   |
| **spacy**        | **>=2.0.11** |
| **ujson**        | **>=1.35**   |

For **reading comprehension module**, please download Spacy language models:

```shell
python -m spacy download en
```

For **detection module**, additional dependencies are required including:

| **gcc**         | **5**      |
| --------------- | ---------- |
| **CMake**       | **3.5**    |
| **cuDNN**       | **7.6**    |
| **blas**        | **1.0**    |
| **mkl**         | **2020.1** |
| **mkl-service** | **2.3.0**  |
| **mkl_fit**     | **1.0**    |
| **mkl_random**  | **1.1**    |

