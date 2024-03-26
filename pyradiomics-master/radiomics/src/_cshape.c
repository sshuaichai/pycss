// 定义宏，告诉NumPy不要使用已弃用的API，而是使用1.7版本的API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// 包含标准库头文件
#include <stdlib.h>

// 包含Python头文件，用于与Python解释器进行交互
#include <Python.h>

// 包含NumPy头文件，用于操作NumPy数组
#include <numpy/arrayobject.h>

// 包含自定义的cshape头文件
#include "cshape.h"

// 模块文档字符串，描述了模块的功能
static char module_docstring[] = "本模块链接到C编译代码，用于pyRadiomics包中高效计算表面积。"
                                 "它通过行进立方体算法提供快速计算，通过\"calculate_surfacearea\"访问。"
                                 "此函数的参数是位置参数，包括两个numpy数组，mask和pixelspacing，必须按此顺序提供。"
                                 "Pixelspacing是一个包含z、y、x维度上像素间距的3元素向量。"
                                 "mask中所有非零元素都被视为分割的一部分，并包含在算法中。";

// 函数文档字符串，描述了函数的参数和功能
static char coefficients_docstring[] = "参数：Mask, PixelSpacing。使用行进立方体算法计算总表面积、体积和最大直径的近似值。"
                                       "等值面被认为位于分割的体素和非分割体素之间的中点。";

// 函数文档字符串，描述了函数的参数和功能
static char coefficients2D_docstring[] = "参数：Mask, PixelSpacing。使用适配的2D行进立方体算法计算总周长、表面和最大直径的近似值。"
                                         "等值面被认为位于分割的像素和非分割像素之间的中点。";

// 声明静态函数原型
static PyObject *cshape_calculate_coefficients(PyObject *self, PyObject *args);
static PyObject *cshape_calculate_coefficients2D(PyObject *self, PyObject *args);

// 检查数组的函数原型，用于验证输入数组
int check_arrays(PyArrayObject *mask_arr, PyArrayObject *spacing_arr, int *size, int *strides, int dimension);

// 模块方法定义
static PyMethodDef module_methods[] = {
  { "calculate_coefficients", cshape_calculate_coefficients, METH_VARARGS, coefficients_docstring },
  { "calculate_coefficients2D", cshape_calculate_coefficients2D, METH_VARARGS, coefficients2D_docstring },
  { NULL, NULL, 0, NULL }
};

#if PY_MAJOR_VERSION >= 3

// Python 3.x模块定义
static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_cshape",           /* m_name */
  module_docstring,    /* m_doc */
  -1,                  /* m_size */
  module_methods,      /* m_methods */
  NULL,                /* m_reload */
  NULL,                /* m_traverse */
  NULL,                /* m_clear */
  NULL,                /* m_free */
};

#endif

// 模块初始化函数
static PyObject *
moduleinit(void)
{
    PyObject *m;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&moduledef);
#else
    m = Py_InitModule3("_cshape",
                       module_methods, module_docstring);
#endif

  if (m == NULL)
      return NULL;

  // 初始化numpy数组功能
  import_array();
  return m;
}

#if PY_MAJOR_VERSION < 3
  PyMODINIT_FUNC
  init_cshape(void)
  {
    moduleinit();
  }
#else
  PyMODINIT_FUNC
  PyInit__cshape(void)
  {
    return moduleinit();
  }
#endif

// 实现cshape_calculate_coefficients函数
static PyObject *cshape_calculate_coefficients(PyObject *self, PyObject *args)
{
  PyObject *mask_obj, *spacing_obj;
  PyArrayObject *mask_arr, *spacing_arr;
  int size[3];
  int strides[3];
  char *mask;
  double *spacing;
  double SA, Volume;
  double diameters[4];
  PyObject *diameter_obj;

  // 解析输入元组
  if (!PyArg_ParseTuple(args, "OO", &mask_obj, &spacing_obj))
    return NULL;

  // 将输入解释为numpy数组
  mask_arr = (PyArrayObject *)PyArray_FROM_OTF(mask_obj, NPY_BYTE, NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY);
  spacing_arr = (PyArrayObject *)PyArray_FROM_OTF(spacing_obj, NPY_DOUBLE, NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY);

  // 检查数组是否满足要求
  if (check_arrays(mask_arr, spacing_arr, size, strides, 3) > 0) return NULL;

  // 获取C类型的数组数据
  mask = (char *)PyArray_DATA(mask_arr);
  spacing = (double *)PyArray_DATA(spacing_arr);

  // 调用C函数计算表面积和体积
  if (calculate_coefficients(mask, size, strides, spacing, &SA, &Volume, diameters))
  {
    // 发生错误，清理并设置异常
    Py_XDECREF(mask_arr);
    Py_XDECREF(spacing_arr);
    PyErr_SetString(PyExc_RuntimeError, "Calculation of Shape coefficients failed.");
    return NULL;
  }

  // 清理并构建结果对象
  Py_XDECREF(mask_arr);
  Py_XDECREF(spacing_arr);

  // 构建直径对象
  diameter_obj = Py_BuildValue("ffff", diameters[0], diameters[1], diameters[2], diameters[3]);

  // 返回结果对象
  return Py_BuildValue("ffN", SA, Volume, diameter_obj);
}

// 实现cshape_calculate_coefficients2D函数
static PyObject *cshape_calculate_coefficients2D(PyObject *self, PyObject *args)
{
  PyObject *mask_obj, *spacing_obj;
  PyArrayObject *mask_arr, *spacing_arr;
  int size[2];
  int strides[2];
  char *mask;
  double *spacing;
  double perimeter, surface, diameter;

  // 解析输入元组
  if (!PyArg_ParseTuple(args, "OO", &mask_obj, &spacing_obj))
    return NULL;

  // 将输入解释为numpy数组
  mask_arr = (PyArrayObject *)PyArray_FROM_OTF(mask_obj, NPY_BYTE, NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY);
  spacing_arr = (PyArrayObject *)PyArray_FROM_OTF(spacing_obj, NPY_DOUBLE, NPY_ARRAY_FORCECAST | NPY_ARRAY_IN_ARRAY);

  // 检查数组是否满足要求
  if (check_arrays(mask_arr, spacing_arr, size, strides, 2) > 0) return NULL;

  // 获取C类型的数组数据
  mask = (char *)PyArray_DATA(mask_arr);
  spacing = (double *)PyArray_DATA(spacing_arr);

  // 调用C函数计算2D表面积和直径
  if (calculate_coefficients2D(mask, size, strides, spacing, &perimeter, &surface, &diameter))
  {
    // 发生错误，清理并设置异常
    Py_XDECREF(mask_arr);
    Py_XDECREF(spacing_arr);
    PyErr_SetString(PyExc_RuntimeError, "Calculation of Shape coefficients failed.");
    return NULL;
  }

  // 清理并构建结果对象
  Py_XDECREF(mask_arr);
  Py_XDECREF(spacing_arr);

  // 构建结果对象
  return Py_BuildValue("fff", perimeter, surface, diameter);
}

// 检查输入数组是否符合要求的函数
int check_arrays(PyArrayObject *mask_arr, PyArrayObject *spacing_arr, int *size, int *strides, int dimension)
{
  int i;

  // 检查输入数组是否为NULL
  if (mask_arr == NULL || spacing_arr == NULL)
  {
    // 发生错误，清理并设置异常
    Py_XDECREF(mask_arr);
    Py_XDECREF(spacing_arr);
    PyErr_SetString(PyExc_RuntimeError, "Error parsing array arguments.");
    return 1;
  }

  // 检查数组维度是否正确
  if (PyArray_NDIM(mask_arr) != dimension || PyArray_NDIM(spacing_arr) != 1)
  {
    // 发生错误，清理并设置异常
    Py_XDECREF(mask_arr);
    Py_XDECREF(spacing_arr);
    PyErr_Format(PyExc_ValueError, "Expected a %iD array for mask, 1D for spacing.", dimension);
    return 2;
  }

  // 检查数组是否是C连续的
  if (!PyArray_IS_C_CONTIGUOUS(mask_arr) || !PyArray_IS_C_CONTIGUOUS(spacing_arr))
  {
    // 发生错误，清理并设置异常
    Py_XDECREF(mask_arr);
    Py_XDECREF(spacing_arr);
    PyErr_SetString(PyExc_ValueError, "Expecting input arrays to be C-contiguous.");
    return 3;
  }

  // 检查spacing数组的维度
  if (PyArray_DIM(spacing_arr, 0) != PyArray_NDIM(mask_arr))
  {
    // 发生错误，清理并设置异常
    Py_XDECREF(mask_arr);
    Py_XDECREF(spacing_arr);
    PyErr_SetString(PyExc_ValueError, "Expecting spacing array to have shape (3,).");
    return 4;
  }

  // 获取mask数组的大小和步幅信息
  for (i = 0; i < dimension; i++)
  {
    size[i] = (int)PyArray_DIM(mask_arr, i);
    strides[i] = (int)(PyArray_STRIDE(mask_arr, i) / PyArray_ITEMSIZE(mask_arr));
  }

  return 0;
}
