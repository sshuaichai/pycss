// 计算三维图像的几何特征和参数
int calculate_coefficients(char *mask, int *size, int *strides, double *spacing,
                           double *surfaceArea, double *volume, double *diameters);
// 计算二维图像的几何特征和参数
int calculate_coefficients2D(char *mask, int *size, int *strides, double *spacing,
                             double *perimeter, double *surface, double *diameter);
