# PyTorchPlus

![PyTorch](https://lh3.googleusercontent.com/KJZpUWcgRNtnJ1CG7nJxhrCLAp-74HUu7W2wJz3DsxeMGQDwSuPlhhsCp2wWx5ReIBZ2YXugsvtj2ckrjUthWmIT6Wy1J1gkk0R-DGOVH18oXzpg6DP-FtpUnh_-t_OQys92eBLo7QyYRtX-MenbgIa5Jcrk9BbDP1kMh7rhiHBCPw2ZbZHyrd4pfzNL2n0VOXxmFfx_ZRkarfCUhqNcsI2w4C1p8yKrRAjPjB08I6E606_HcizOayodaKXrbiBzGV0uel7ludtgM1Hv88OnzrOzLPXeajjrIPeyOOwi2io61gb7GiAZN7BA4v1aU9IYHdf6DQnDjQAOxh84RcMrUXH2ZuH9Enk-0sNSoyHxpurDWqko9EJKja5cvWBmwf5oq6hW8SqIKAV4EUBa7cYorZczOFnv36EN0Y1Pt9zsodZGJr1VV8ShvGvlZR6PMEideqaOnMsRvx5Ws4vHwdSra45EhKNo3S_gipD2Ld8qe8u_rOTHSwKUwUuPFetO646Wcmg483UDP-68HtGVhvRDoTxAExlXIwwjlozJiv7jLu-XVYj4U7rzGYo4Xv-NKe0nR5zAaAV31NLBQm0eHB9Tfic6BnyCjnsJzOYWX-LAK38pPTLKbWnIlccXkR1gyjD5oT3j4pEWnNV-x6AdpA0f62nzQ12FAitJ0gQIiuU9-se4YbNmPvntqGetvG_n9w=s400-no?authuser=0)

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

