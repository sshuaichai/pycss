#!/usr/bin/env python
import argparse  # 导入解析命令行参数的模块
import csv  # 导入CSV文件操作的模块
from functools import partial  # 导入函数部分应用的模块
import json  # 导入JSON操作的模块
import logging.config  # 导入日志配置模块
import logging.handlers  # 导入日志处理模块
from multiprocessing import cpu_count, Manager, Pool  # 导入多进程相关模块
import os  # 导入操作系统接口模块
import sys  # 导入系统相关的参数和函数模块
import threading  # 导入线程操作的模块

import numpy  # 导入NumPy模块，用于数学运算
from pykwalify.compat import yaml  # 导入兼容YAML操作的模块
import pykwalify.core  # 导入核心验证库
import six.moves  # 导入兼容Python 2和3的模块

import radiomics  # 导入radiomics特征提取库
import radiomics.featureextractor  # 导入特征提取器模块
from . import segment, voxel  # 导入分割和体素处理模块

from ruamel.yaml import YAML  # 导入YAML处理库

class PyRadiomicsCommandLine:
  # 定义PyRadiomics命令行界面类

  def __init__(self, custom_arguments=None):
    # 类初始化函数
    self.logger = logging.getLogger('radiomics.script')  # 初始化日志记录器
    self.relative_path_start = os.getcwd()  # 获取当前工作目录
    self.args = self.getParser().parse_args(args=custom_arguments)  # 解析命令行参数

    self.logging_config, self.queue_listener = self._configureLogging()  # 配置日志

    # 根据模式选择对应的函数
    if self.args.mode == 'segment':
      self.serial_func = segment.extractSegment  # 单线程分割函数
      self.parallel_func = segment.extractSegment_parallel  # 多线程分割函数
    else:
      self.serial_func = voxel.extractVoxel  # 单线程体素函数
      self.parallel_func = voxel.extractVoxel_parallel  # 多线程体素函数

    self.case_count = 0  # 初始化案例计数
    self.num_workers = 0  # 初始化工作线程数

  @classmethod
  def getParser(cls):
    # 定义解析命令行参数的函数
    parser = argparse.ArgumentParser(usage='%(prog)s image|batch [mask] [Options]',
                                     formatter_class=argparse.RawTextHelpFormatter)

    # 定义输入参数组
    inputGroup = parser.add_argument_group(title='输入',
                                           description='定义提取的输入文件和参数：\n'
                                                       '- 图像和掩码文件（单模式）'
                                                       '或指定它们的CSV文件（批处理模式）\n'
                                                       '- 参数文件（.yml/.yaml或.json）\n'
                                                       '- 自定义类型3（"settings"）的覆盖\n'
                                                       '- 多线程批处理')
    inputGroup.add_argument('input', metavar='{Image,Batch}FILE',
                            help='图像文件（单模式）或CSV批处理文件（批处理模式）')
    inputGroup.add_argument('mask', nargs='?', metavar='MaskFILE', default=None,
                            help='标识图像中ROI的掩码文件。\n'
                                 '仅在单模式下需要，否则忽略。')
    inputGroup.add_argument('--param', '-p', metavar='FILE', default=None,
                            help='包含提取中使用的设置的参数文件')
    inputGroup.add_argument('--setting', '-s', metavar='"SETTING_NAME:VALUE"', action='append', default=[], type=str,
                            help='将覆盖参数文件中的额外参数\n'
                                 '和/或默认设置。可能有多个\n'
                                 '设置。注意：仅适用于自定义\n'
                                 '类型3（"setting"）。')
    inputGroup.add_argument('--jobs', '-j', metavar='N', type=int, default=1,
                            choices=six.moves.range(1, cpu_count() + 1),
                            help='（仅限批处理模式）指定用于\n'
                                 '并行处理的线程数。这是在案例级别应用的；\n'
                                 '即每个案例1个线程。实际使用的工作数为\n'
                                 'min(cases, jobs)。')
    inputGroup.add_argument('--validate', action='store_true',
                            help='如果指定，检查输入是否有效并检查文件位置是否指向存在的'
                                 '文件')

    # 定义输出参数组
    outputGroup = parser.add_argument_group(title='输出', description='控制输出重定向和'
                                                                        '计算结果的格式化的参数。')
    outputGroup.add_argument('--out', '-o', metavar='FILE', type=argparse.FileType('a'), default=sys.stdout,
                             help='追加输出的文件。')
    outputGroup.add_argument('--out-dir', '-od', type=str, default=None,
                             help='存储输出的目录。如果在分段模式下指定，这将为'
                                  '每个处理的案例写入csv输出。在体素模式下，此目录用于存储特征图。'
                                  '如果在体素模式下未指定，则使用当前工作目录。')
    outputGroup.add_argument('--mode', '-m', choices=['segment', 'voxel'], default='segment',
                             help='PyRadiomics的提取模式。')
    outputGroup.add_argument('--skip-nans', action='store_true',
                             help='添加此参数以跳过返回具有\n'
                                  '无效结果（NaN）的特征')
    outputGroup.add_argument('--format', '-f', choices=['csv', 'json', 'txt'], default='txt',
                             help='输出的格式。\n'
                                  '"txt"（默认）：每行一个特征，格式为"case-N_name:value"\n'
                                  '"json"：特征以JSON格式字典写入\n'
                                  '（每个案例1个字典，每行1个案例）"{name:value}"\n'
                                  '"csv"：一行特征名，然后每个案例一行\n'
                                  '特征值。')
    outputGroup.add_argument('--format-path', choices=['absolute', 'relative', 'basename'], default='absolute',
                             help='控制输出中的输入图像和掩码路径格式。\n'
                                  '"absolute"（默认）：绝对文件路径。\n'
                                  '"relative"：相对于当前工作目录的文件路径。\n'
                                  '"basename"：仅存储文件名。')
    outputGroup.add_argument('--unix-path', '-up', action='store_true',
                             help='如果指定，确保输出中的所有路径\n'
                                  '使用unix风格的路径分隔符（"/"）。')

    # 定义日志参数组
    loggingGroup = parser.add_argument_group(title='日志',
                                             description='控制输出到控制台和（可选）日志文件的'
                                                         '（日志量）。')
    loggingGroup.add_argument('--logging-level', metavar='LEVEL',
                              choices=['NOTSET', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                              default='WARNING', help='设置日志捕获级别')
    loggingGroup.add_argument('--log-file', metavar='FILE', default=None, help='追加日志输出到的文件')
    loggingGroup.add_argument('--verbosity', '-v', action='store', nargs='?', default=3, const=4, type=int,
                              choices=[1, 2, 3, 4, 5],
                              help='调节输出到stderr的量。默认[3]，打印\n'
                                   '级别WARNING及以上的日志。通过指定此\n'
                                   '参数而不提供值，默认为级别INFO [4]。\n'
                                   '更高的值导致更详细的输出。')
    parser.add_argument('--label', '-l', metavar='N', default=None, type=int,
                        help='(已弃用) 用于\n'
                             '特征提取的掩码中的标签值。')

    parser.add_argument('--version', action='version', help='打印版本并退出',
                        version='%(prog)s ' + radiomics.__version__)
    return parser

  def run(self):
    # 运行提取过程
    try:
      self.logger.info('开始PyRadiomics（版本：%s）', radiomics.__version__)
      caseGenerator = self._processInput()  # 处理输入
      if caseGenerator is not None:
        if self.args.validate:
          self._validateCases(caseGenerator)  # 验证案例
        else:
          results = self._processCases(caseGenerator)  # 处理案例
          self._processOutput(results)  # 处理输出
          self.logger.info('成功完成基于%s的提取...', self.args.mode)
      else:
        return 1  # 特征提取错误
    except (KeyboardInterrupt, SystemExit):
      self.logger.info('取消提取')
      return -1
    except Exception:
      self.logger.error('提取特征时出错！', exc_info=True)
      return 3  # 未知错误
    finally:
      if self.queue_listener is not None:
        self.queue_listener.stop()  # 停止日志监听器
    return 0  # 成功

  def _processInput(self):
    # 处理输入数据
    self.logger.info('处理输入...')

    self.case_count = 1
    self.num_workers = 1

    # 检查输入是否为批处理文件
    if self.args.input.endswith('.csv'):
      self.logger.debug('加载批处理文件“%s”', self.args.input)
      self.relative_path_start = os.path.dirname(self.args.input)
      with open(self.args.input, mode='r') as batchFile:
        cr = csv.DictReader(batchFile, lineterminator='\n')

        # 检查是否存在必需的图像和掩码列
        if 'Image' not in cr.fieldnames:
          self.logger.error('输入中缺少必需的“Image”列，无法提取特征...')
          return None
        if 'Mask' not in cr.fieldnames:
          self.logger.error('输入中缺少必需的“Mask”列，无法提取特征...')
          return None

        cases = []
        for row_idx, row in enumerate(cr, start=2):
          if row['Image'] is None or row['Mask'] is None:
            self.logger.warning('批处理L%d：缺少必需的图像或掩码，跳过此案例...', row_idx)
            continue
          imPath = row['Image']
          maPath = row['Mask']
          if not os.path.isabs(imPath):
            imPath = os.path.abspath(os.path.join(self.relative_path_start, imPath))
            self.logger.debug('认为图像文件路径相对于输入CSV。绝对路径：%s', imPath)
          if not os.path.isabs(maPath):
            maPath = os.path.abspath(os.path.join(self.relative_path_start, maPath))
            self.logger.debug('认为掩码文件路径相对于输入CSV。绝对路径：%s', maPath)
          cases.append(row)
          cases[-1]['Image'] = imPath
          cases[-1]['Mask'] = maPath

          self.case_count = len(cases)
        caseGenerator = enumerate(cases, start=1)
        self.num_workers = min(self.case_count, self.args.jobs)
    elif self.args.mask is not None:
      caseGenerator = [(1, {'Image': self.args.input, 'Mask': self.args.mask})]
    else:
      self.logger.error('输入未被识别为批处理，未指定掩码，无法计算结果！')
      return None

    return caseGenerator

  def _validateCases(self, case_generator):
    # 验证案例
    self.logger.info('为%i个案例验证输入', self.case_count)
    errored_cases = 0
    for case_idx, case in case_generator:
      if case_idx == 1 and self.args.param is not None:
        if not os.path.isfile(self.args.param):
          self.logger.error('指定的参数文件路径不存在！')
        else:
          schemaFile, schemaFuncs = radiomics.getParameterValidationFiles()

          c = pykwalify.core.Core(source_file=self.args.param, schema_files=[schemaFile], extensions=[schemaFuncs])
          try:
            c.validate()
          except (KeyboardInterrupt, SystemExit):
            raise
          except Exception:
            self.logger.error('参数验证失败！', exc_info=True)
            self.logger.debug("验证案例（%i/%i）：%s", case_idx, self.case_count, case)

      case_error = False
      if not os.path.isfile(case['Image']):
        case_error = True
        self.logger.error('案例（%i/%i）的图像路径不存在！', case_idx, self.case_count)
      if not os.path.isfile(case['Mask']):
        case_error = True
        self.logger.error('案例（%i/%i）的掩码路径不存在！', case_idx, self.case_count)

      if case_error:
        errored_cases += 1

    self.logger.info('验证完成，%i个案例中发现错误', errored_cases)

  def _processCases(self, case_generator):
    # 处理案例
    setting_overrides = self._parseOverrides()  # 解析覆盖设置

    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(self.args.param, **setting_overrides)

    if self.args.out_dir is not None and not os.path.isdir(self.args.out_dir):
      os.makedirs(self.args.out_dir)

    if self.num_workers > 1:  # 多个案例，启用并行处理
      self.logger.info('输入有效，开始从%d个案例中并行提取，使用%d个工作线程...',
                       self.case_count, self.num_workers)
      pool = Pool(self.num_workers)
      try:
        task = pool.map_async(partial(self.parallel_func,
                                      extractor=extractor,
                                      out_dir=self.args.out_dir,
                                      logging_config=self.logging_config,
                                      unix_path=self.args.unix_path),
                              case_generator,
                              chunksize=min(10, self.case_count))
        # 等待结果完成。task.get()没有超时会执行阻塞调用，这阻止了程序处理中断信号
        while not task.ready():
          pass
        results = task.get()
        pool.close()
      except (KeyboardInterrupt, SystemExit):
        pool.terminate()
        raise
      finally:
        pool.join()
    elif self.num_workers == 1:  # 单个案例或顺序批处理
      self.logger.info('输入有效，开始从%d个案例中顺序提取...',
                       self.case_count)
      results = []
      for case in case_generator:
        results.append(self.serial_func(*case,
                                        extractor=extractor,
                                        out_dir=self.args.out_dir,
                                        unix_path=self.args.unix_path))
    else:
      # 批处理中未定义案例
      self.logger.error('没有要处理的案例...')
      results = None
    return results

  def _processOutput(self, results):
    # 处理输出结果
    self.logger.info('处理结果...')

    # 存储所有计算特征的头部
    # 通过检查案例>1的所有头部，并减去案例1中已有的头部，保留案例1的原始排序
    # 附加头部默认由pyradiomics生成，因此可以在末尾追加
    additional_headers = set()
    for case in results[1:]:
      additional_headers.update(set(case.keys()))
      additional_headers -= set(results[0].keys())  # 减去第一个案例中找到的所有头部

    headers = list(results[0].keys()) + sorted(additional_headers)

    # 设置图像和掩码路径的格式化规则
    if self.args.format_path == 'absolute':
      pathFormatter = os.path.abspath
    elif self.args.format_path == 'relative':
      pathFormatter = partial(os.path.relpath, start=self.relative_path_start)
    elif self.args.format_path == 'basename':
      pathFormatter = os.path.basename
    else:
      self.logger.warning('未识别的路径格式（%s），恢复为默认（"absolute"）',
                          self.args.format_path)
      pathFormatter = os.path.abspath

    for case_idx, case in enumerate(results, start=1):
      # 如果指定，跳过NaN值
      if self.args.skip_nans:
        for key in list(case.keys()):
          if isinstance(case[key], float) and numpy.isnan(case[key]):
            self.logger.debug('案例%d，特征%s计算为NaN，从结果中移除', case_idx, key)
            del case[key]

      # 格式化图像和掩码文件的路径
      case['Image'] = pathFormatter(case['Image'])
      case['Mask'] = pathFormatter(case['Mask'])

      if self.args.unix_path and os.path.sep != '/':
        case['Image'] = case['Image'].replace(os.path.sep, '/')
        case['Mask'] = case['Mask'].replace(os.path.sep, '/')

      # 如果格式为'csv'或'txt'，则按案例写出结果，'json'格式在此循环外处理（问题#483）
      if self.args.format == 'csv':
        writer = csv.DictWriter(self.args.out, headers, lineterminator='\n', extrasaction='ignore')
        if case_idx == 1:
          writer.writeheader()
        writer.writerow(case)  # 如果启用skip_nans，nan值被写为空字符串
      elif self.args.format == 'txt':
        for k, v in six.iteritems(case):
          self.args.out.write('案例-%d_%s: %s\n' % (case_idx, k, v))

    # JSON案例的转储在循环外处理，否则结果文档将无效
    if self.args.format == 'json':
      # JSON无法序列化numpy数组，即使该数组表示标量值（PyRadiomics特征值）
      # 因此，使用此编码器，它首先将numpy数组转换为python列表，这些列表是JSON可序列化的
      class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
          if isinstance(obj, numpy.ndarray):
            return obj.tolist()
          return json.JSONEncoder.default(self, obj)

      json.dump(results, self.args.out, cls=NumpyEncoder, indent=2)

  def _parseOverrides(self):
    # 解析覆盖设置
    setting_overrides = {}

    # 解析覆盖
    if len(self.args.setting) == 0:
      self.logger.debug('未找到覆盖')
      return setting_overrides

    self.logger.debug('读取参数模式')
    schemaFile, schemaFuncs = radiomics.getParameterValidationFiles()
    with open(schemaFile) as schema:
      yaml = YAML(typ='safe', pure=True)
      settingsSchema = yaml.load(schema)['mapping']['setting']['mapping']

    # 解析单个值函数
    def parse_value(value, value_type):
      if value_type == 'str':
        return value  # 无转换
      elif value_type == 'int':
        return int(value)
      elif value_type == 'float':
        return float(value)
      elif value_type == 'bool':
        return value == '1' or value.lower() == 'true'
      else:
        raise ValueError('无法理解的value_type "%s"' % value_type)

    for setting in self.args.setting:  # setting = "setting_key:setting_value"
      if ':' not in setting:
        self.logger.warning('覆盖设置"%s"的格式不正确，缺少":"', setting)
        continue
      # 分割为键和值
      setting_key, setting_value = setting.split(':', 2)

      # 检查它是否是有效的PyRadiomics设置
      if setting_key not in settingsSchema:
        self.logger.warning('未识别覆盖"%s"，跳过...', setting_key)
        continue

      # 尝试通过查找settingsSchema中的类型来解析值
      try:
        setting_def = settingsSchema[setting_key]
        setting_type = 'str'  # 如果模式中省略了类型，将其视为字符串（无转换）
        if 'seq' in setting_def:
          # 多值设置
          if len(setting_def['seq']) > 0 and 'type' in setting_def['seq'][0]:
            setting_type = setting_def['seq'][0]['type']

          setting_overrides[setting_key] = [parse_value(val, setting_type) for val in setting_value.split(',')]
          self.logger.debug('将"%s"解析为列表（元素类型"%s"）；值：%s',
                            setting_key, setting_type, setting_overrides[setting_key])
        else:
          if 'type' in setting_def:
            setting_type = setting_def['type']
          setting_overrides[setting_key] = parse_value(setting_value, setting_type)
          self.logger.debug('将"%s"解析为类型"%s"；值：%s', setting_key, setting_type,
                            setting_overrides[setting_key])

      except (KeyboardInterrupt, SystemExit):
        raise
      except Exception:
        self.logger.warning('无法解析设置"%s"的值"%s"，跳过...', setting_value, setting_key)

    # 弃用参数label的部分
    if self.args.label is not None:
      self.logger.warning(
        '参数"label"已弃用。要指定自定义标签，请使用"setting"参数，如下：'
        '"--setting=label:N"，其中N是标签值。')
      setting_overrides['label'] = self.args.label
    # 结束弃用部分

    return setting_overrides

  def _configureLogging(self):
    # 配置日志处理
    # 在多进程情况下处理子进程的日志消息的监听器
    queue_listener = None

    logfileLevel = getattr(logging, self.args.logging_level)
    verboseLevel = (6 - self.args.verbosity) * 10  # 转换为python日志级别
    logger_level = min(logfileLevel, verboseLevel)

    logging_config = {
      'version': 1,
      'disable_existing_loggers': False,
      'formatters': {
        'default': {
          'format': '[%(asctime)s] %(levelname)-.1s: %(name)s: %(message)s',
          'datefmt': '%Y-%m-%d %H:%M:%S'
        }
      },
      'handlers': {
        'console': {
          'class': 'logging.StreamHandler',
          'level': verboseLevel,
          'formatter': 'default'
        }
      },
      'loggers': {
        'radiomics': {
          'level': logger_level,
          'handlers': ['console']
        }
      }
    }

    if self.args.jobs > 1:
      # 如果启用了多进程，更新日志格式以包括线程名
      logging_config['formatters']['default']['format'] = \
        '[%(asctime)s] %(levelname)-.1s: (%(threadName)s) %(name)s: %(message)s'

    # 设置可选的日志文件记录
    if self.args.log_file is not None:
      py_version = (sys.version_info.major, sys.version_info.minor)
      if self.args.jobs > 1 and py_version >= (3, 2):
        # 多进程！使用QueueHandler、FileHandler和QueueListener
        # 实现线程安全日志。

        # 但是，QueueHandler和Listener是在python 3.2中添加的。
        # 因此，只有当python版本>3.2时才使用这个
        q = Manager().Queue(-1)
        threading.current_thread().setName('Main')

        logging_config['handlers']['logfile'] = {
          'class': 'logging.handlers.QueueHandler',
          'queue': q,
          'level': logfileLevel,
          'formatter': 'default'
        }

        file_handler = logging.FileHandler(filename=self.args.log_file, mode='a')
        file_handler.setFormatter(logging.Formatter(fmt=logging_config['formatters']['default'].get('format'),
                                                    datefmt=logging_config['formatters']['default'].get('datefmt')))

        queue_listener = logging.handlers.QueueListener(q, file_handler)
        queue_listener.start()
      else:
        logging_config['handlers']['logfile'] = {
          'class': 'logging.FileHandler',
          'filename': self.args.log_file,
          'mode': 'a',
          'level': logfileLevel,
          'formatter': 'default'
        }
      logging_config['loggers']['radiomics']['handlers'].append('logfile')

    logging.config.dictConfig(logging_config)

    self.logger.debug('日志初始化')
    return logging_config, queue_listener


def parse_args():
  # 解析命令行参数的函数
  try:
    return PyRadiomicsCommandLine().run()
  except Exception as e:
    logging.getLogger().error("执行PyRadiomics命令行时出错！", exc_info=True)
    print("执行PyRadiomics命令行时出错！\n%s" % e)
    return 4
