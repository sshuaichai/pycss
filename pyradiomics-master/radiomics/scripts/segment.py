from collections import OrderedDict  # 有序字典
import csv  # CSV文件操作
from datetime import datetime  # 日期和时间操作
import logging.config  # 日志配置
import os  # 操作系统接口
import threading  # 线程操作

import SimpleITK as sitk  # SimpleITK库，用于医学图像处理
import six  # Python 2和3兼容性库

import radiomics  # Radiomics特征提取库

caseLogger = logging.getLogger('radiomics.script')  # 获取日志记录器
_parallel_extraction_configured = False  # 标记是否已配置并行提取


def extractSegment(case_idx, case, extractor, **kwargs):
  global caseLogger  # 使用全局日志记录器

  out_dir = kwargs.get('out_dir', None)  # 获取输出目录

  if out_dir is None:
    return _extractFeatures(case_idx, case, extractor)  # 如果未指定输出目录，直接提取特征

  filename = os.path.join(out_dir, 'features_%s.csv' % case_idx)  # 构造输出文件名
  if os.path.isfile(filename):
    # 如果输出文件已存在，读取结果（防止中断过程中重新提取）
    with open(filename, 'r') as outputFile:
      reader = csv.reader(outputFile)
      headers = six.next(reader)  # 读取表头
      values = six.next(reader)  # 读取值
      feature_vector = OrderedDict(zip(headers, values))  # 创建有序字典存储特征向量

    caseLogger.info('Patient %s already processed, reading results...', case_idx)
  else:
    # 提取特征集
    feature_vector = _extractFeatures(case_idx, case, extractor)

    # 将结果存储在临时文件中，避免写入冲突
    with open(filename, 'w') as outputFile:
      writer = csv.DictWriter(outputFile, fieldnames=list(feature_vector.keys()), lineterminator='\n')
      writer.writeheader()
      writer.writerow(feature_vector)

  return feature_vector  # 返回特征向量


def _extractFeatures(case_idx, case, extractor):
  global caseLogger  # 使用全局日志记录器

  feature_vector = OrderedDict(case)  # 实例化输出的特征向量

  try:
    caseLogger.info('Processing case %s', case_idx)
    t = datetime.now()  # 记录开始时间

    # 从案例中获取必要的文件路径和标签
    imageFilepath = case['Image']  # 必需
    maskFilepath = case['Mask']  # 必需
    label = case.get('Label', None)  # 可选
    if isinstance(label, six.string_types):
      label = int(label)
    label_channel = case.get('Label_channel', None)  # 可选
    if isinstance(label_channel, six.string_types):
      label_channel = int(label_channel)

    # 提取特征
    feature_vector.update(extractor.execute(imageFilepath, maskFilepath, label, label_channel))

    delta_t = datetime.now() - t  # 计算处理时间
    caseLogger.info('Case %s processed in %s', case_idx, delta_t)

  except (KeyboardInterrupt, SystemExit):  # 处理中断错误
    raise
  except SystemError:  # 处理SimpleITK调用中的中断
    raise KeyboardInterrupt()
  except Exception:
    caseLogger.error('Feature extraction failed!', exc_info=True)

  return feature_vector  # 返回特征向量


def extractSegment_parallel(args, logging_config=None, **kwargs):
  try:
    threading.current_thread().name = 'case %s' % args[0]  # 设置线程名称为案例索引

    if logging_config is not None:
      _configureParallelExtraction(logging_config)  # 配置并行提取日志

    return extractSegment(*args, **kwargs)
  except (KeyboardInterrupt, SystemExit):
    return None  # 处理中断，返回None


def _configureParallelExtraction(logging_config, add_info_filter=True):
  """
  初始化并行提取的日志。这需要在这里完成，因为它需要为创建的每个线程完成。
  """
  global _parallel_extraction_configured
  if _parallel_extraction_configured:
    return

  logging.config.dictConfig(logging_config)  # 配置日志

  if add_info_filter:
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

    outputhandler = radiomics.logger.handlers[0]  # 获取输出处理器
    outputhandler.addFilter(info_filter('radiomics.script'))  # 添加过滤器

  sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)  # 确保每个案例的提取在一个线程上处理

  _parallel_extraction_configured = True
  radiomics.logger.debug('parallel extraction configured')  # 记录并行提取配置完成
