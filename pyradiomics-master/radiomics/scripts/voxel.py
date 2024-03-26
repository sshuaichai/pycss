from collections import OrderedDict  # 导入有序字典类
from datetime import datetime  # 导入日期时间类
import logging.config  # 导入日志配置模块
import os  # 导入操作系统接口模块
import threading  # 导入线程操作模块

import SimpleITK as sitk  # 导入SimpleITK库，用于医学图像处理
import six  # 导入兼容Python 2和3的模块

import radiomics  # 导入radiomics特征提取库

caseLogger = logging.getLogger('radiomics.script')  # 初始化案例日志记录器
_parallel_extraction_configured = False  # 初始化并行提取配置标志

def extractVoxel(case_idx, case, extractor, **kwargs):
  # 定义体素级特征提取函数
  global caseLogger

  out_dir = kwargs.get('out_dir', None)  # 获取输出目录参数，默认为None
  unix_path = kwargs.get('unix_path', False)  # 获取Unix路径风格参数，默认为False

  # 实例化输出特征向量
  feature_vector = OrderedDict(case)

  try:
    if out_dir is None:
      out_dir = '.'  # 如果未指定输出目录，则使用当前目录
    elif not os.path.isdir(out_dir):
      caseLogger.debug('Creating output directory at %s' % out_dir)
      os.makedirs(out_dir)  # 如果输出目录不存在，则创建

    caseLogger.info('Processing case %s', case_idx)  # 记录处理案例信息
    t = datetime.now()  # 记录开始时间

    imageFilepath = case['Image']  # 获取必需的图像文件路径
    maskFilepath = case['Mask']  # 获取必需的掩码文件路径
    label = case.get('Label', None)  # 获取可选的标签，默认为None
    if isinstance(label, six.string_types):
      label = int(label)  # 如果标签是字符串类型，则转换为整数
    label_channel = case.get('Label_channel', None)  # 获取可选的标签通道，默认为None
    if isinstance(label_channel, six.string_types):
      label_channel = int(label_channel)  # 如果标签通道是字符串类型，则转换为整数

    # 执行特征提取
    result = extractor.execute(imageFilepath, maskFilepath, label, label_channel, voxelBased=True)

    for k in result:
      if isinstance(result[k], sitk.Image):
        # 如果结果是SimpleITK图像，则保存为nrrd文件
        target = os.path.join(out_dir, 'Case-%i_%s.nrrd' % (case_idx, k))
        sitk.WriteImage(result[k], target, True)
        if unix_path and os.path.sep != '/':
          target = target.replace(os.path.sep, '/')  # 如果需要Unix路径风格，则替换路径分隔符
        feature_vector[k] = target
      else:
        feature_vector[k] = result[k]  # 否则直接保存结果值

    # 显示处理消息
    delta_t = datetime.now() - t  # 计算处理时间
    caseLogger.info('Case %s processed in %s', case_idx, delta_t)  # 记录处理时间

  except (KeyboardInterrupt, SystemExit):  # 捕获中断提取的错误
    raise
  except SystemError:
    # 当在处理SimpleITK调用时捕获键盘中断时发生
    raise KeyboardInterrupt()
  except Exception:
    caseLogger.error('Feature extraction failed!', exc_info=True)  # 记录特征提取失败的错误

  return feature_vector  # 返回特征向量

def extractVoxel_parallel(args, logging_config=None, **kwargs):
  # 定义并行体素级特征提取函数
  try:
    # 将线程名设置为案例名称
    threading.current_thread().name = 'case %s' % args[0]  # args[0] = case_idx

    if logging_config is not None:
      _configureParallelExtraction(logging_config)  # 配置并行提取日志

    return extractVoxel(*args, **kwargs)  # 调用体素级特征提取函数
  except (KeyboardInterrupt, SystemExit):
    # 在这里捕获错误，因为这代表子进程的中断。
    # 主进程也被中断，取消操作在那里进一步处理
    return None

def _configureParallelExtraction(logging_config, add_info_filter=True):
  """
  为并行提取初始化日志。这需要在这里完成，因为它需要为创建的每个线程完成。
  """
  global _parallel_extraction_configured
  if _parallel_extraction_configured:
    return  # 如果已配置，则直接返回

  # 配置日志
  ###################

  logging.config.dictConfig(logging_config)  # 应用日志配置

  if add_info_filter:
    # 定义过滤器，允许来自指定过滤器和级别INFO及以上的消息，以及来自其他记录器的级别WARNING及以上的消息。
    class info_filter(logging.Filter):
      def __init__(self, name):
        super(info_filter, self).__init__(name)
        self.level = logging.WARNING

      def filter(self, record):
        if record.levelno >= self.level:
          return True
        if record.name == self.name and record.levelno >= logging.INFO:
          return True
        return False

    # 将过滤器添加到radiomics记录器的第一个处理程序限制输出中的信息消息仅来自radiomics.script，但整个库的警告和错误也打印到输出。
    # 这不影响存储在日志文件中的日志量。
    outputhandler = radiomics.logger.handlers[0]  # 输出到输出的处理程序
    outputhandler.addFilter(info_filter('radiomics.script'))

  # 确保每个案例的整个提取在1个线程上处理
  ####################################################################

  sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)  # 设置SimpleITK全局默认线程数为1

  _parallel_extraction_configured = True
  radiomics.logger.debug('parallel extraction configured')  # 记录并行提取配置完成
