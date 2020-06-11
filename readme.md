# PyTorchPlus

![PyTorch](https://lh3.googleusercontent.com/mTBKsJYq_C79xnNY80W12JhOs7rLN2t8PjfQq7A7JPgZm9f7AeReyjVcCT3AyeF5fidl1-nDpn-r9oU2IEabDS0qMXEGCLOr24LD2TDbFD1OflnSLIdYjSdsKOb3BlCEeuxc2KICOwY4imr26Fdt-aU2hbv4Kppdl940Y8K2vG3p-wsEKuPmTR0VqF2YyZ9ZzTq1qS6QU0Hi6UYgcOPKoFGso2H8ty1yANdiD4mGMiwfJTr2SF3GOn-2DtFcEv72mg-8ETku_tYadsaO18hk1CVzUmimad5y3QVqI6E0ft4MfHi1o71rt0JTvs9hq6PRFbIDBdGJ6CK1dD-IMc_eGgP65LG5E8DUC1LlgVzqpbBj2kLUeluXJNj56p6uCO6WLmCl_MKk3lIMb5wLlfpCbruyKzucLDFHWS0GiWynwq3oHVXlngqw6ZEzoIFoLJOPz3hxCx75vzBj7uO6BgV72DIV7L2bH0udCUjcfbG-ZUlR1TMwa37KVxXZX2HGLjWrdiV4j94BCcU6srbuvjO6a2FUarr-5D0HlAiIOfZpPaWxHP2rQvoPYYIj9Pg8-8ZjxzOGsLrdBWsgVqP1gFDWvjOl1j6eKlzNw8nP3xj-u9YvuBoTB4xr0I-zE9yPDGrDHDSHbOAfau-ImoSgJ1OioZEm1YeBaKOJhRdrAxD7cwOA23QhpAv22ZykAT8tsw=s400-no?authuser=0)

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

