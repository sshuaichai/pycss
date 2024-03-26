# 为了方便起见，将collection和numpy导入“pyradiomics”命名空间

import collections  # 不导入只是为了方便，在注释中声明
import inspect
import logging
import os
import pkgutil
import sys
import tempfile

import numpy  # 不导入只是为了方便，在注释中声明
from six.moves import urllib

from . import imageoperations


def deprecated(func):
  """
  装饰器函数用于标记函数为已弃用。这用于确保已弃用的特性函数在启用“全部”特性时不会添加到启用的特性列表中。
  """
  func._is_deprecated = True
  return func


def setVerbosity(level):
  """
  更改PyRadiomics在提取期间应打印的信息量。级别越低，打印到输出（stderr）的信息越多。

  使用“level”（Python定义的日志级别）参数，可以选择以下级别：

  - 60：安静模式，不打印任何消息到stderr
  - 50：仅打印级别为“CRITICAL”的日志消息
  - 40：打印级别为“ERROR”及更高级别的日志消息
  - 30：打印级别为“WARNING”及更高级别的日志消息
  - 20：打印级别为“INFO”及更高级别的日志消息
  - 10：打印级别为“DEBUG”及更高级别的日志消息（即所有日志消息）

  默认情况下，radiomics日志记录器的级别设置为“INFO”，stderr处理程序的级别设置为“WARNING”。因此，可以通过向radiomics日志记录器添加适当的处理程序，轻松设置一个存储从级别“INFO”及更高级别的提取日志消息的日志，而stderr输出仍然仅包含警告和错误。

  .. 注意::

    此函数假定在工具箱初始化时添加到radiomics日志记录器的处理程序没有从日志记录器处理程序中删除，并因此保持为第一个处理程序。

  .. 注意::

    这不会影响日志记录器本身的级别（例如，如果verbosity level = 3，则日志级别为DEBUG的日志消息仍然可以存储在日志文件中，如果将适当的处理程序添加到日志记录器并将日志记录器的日志级别设置为正确级别）。*例外情况：如果verbosity设置为DEBUG，则日志记录器的级别也会降低为DEBUG。如果然后再次提高verbosity级别，日志记录器级别将保持为DEBUG。*
  """
  global logger, handler
  if level < 10:  # 最低级别：DEBUG
    level = 10
  if level > 60:  # 最高级别= 50（CRITICAL），级别60会导致“安静”模式
    level = 60

  handler.setLevel(level)
  if handler.level < logger.level:  # 如果必要，降低日志记录器的级别
    logger.setLevel(level)


def getFeatureClasses():
  """
  枚举radiomics包的所有模块，使用pkgutil导入这些模块。

  返回包含特征类的所有模块的字典，以模块名称作为键，以特征类的抽象类对象作为值。假定每个模块只包含一个特征类

  这是通过inspect.getmembers实现的。仅在其包含一个成员是类的模块被添加，成员名称以'Radiomics'开头，并且是从:py:class:`radiomics.base.RadiomicsFeaturesBase`继承的类。

  这个迭代只在一次（在工具箱初始化时）运行，随后的调用返回第一次调用创建的字典。
  """
  global _featureClasses
  if _featureClasses is None:  # 第一次调用时，枚举可能的特征类并导入PyRadiomics模块
    _featureClasses = {}
    for _, mod, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
      if str(mod).startswith('_'):  # 跳过加载'私有'类，这些类不包含特征类
        continue
      __import__('radiomics.' + mod)
      module = sys.modules['radiomics.' + mod]
      attributes = inspect.getmembers(module, inspect.isclass)
      for a in attributes:
        if a[0].startswith('Radiomics'):
          for parentClass in inspect.getmro(a[1])[1:]:  # 仅包括从RadiomicsFeaturesBase继承的类
            if parentClass.__name__ == 'RadiomicsFeaturesBase':
              _featureClasses[mod] = a[1]
              break

  return _featureClasses


def getImageTypes():
  """
  返回可能的图像类型列表（即可能的滤波器和“原始”未经滤波的图像类型）。此函数通过将签名（“get<imageType>Image”）与:ref:`imageoperations <radiomics-imageoperations-label>`中定义的函数进行匹配，动态地查找图像类型。返回一个包含可用图像类型名称（相应函数名称的<imageType>部分）的列表。

  这个迭代只在工具箱初始化时运行一次。找到的结果会存储并在后续调用时返回。
  """
  global _imageTypes
  if _imageTypes is None:  # 第一次调用时，枚举可能的输入图像类型（原始和任何滤波器）
    _imageTypes = [member[3:-5] for member in dir(imageoperations)
                   if member.startswith('get') and member.endswith("Image")]

  return _imageTypes


def getTestCase(testCase, dataDirectory=None):
  """
  此函数为测试PyRadiomics提供了一个图像和掩码。可以选择七个测试案例中的一个：

   - brain1
   - brain2
   - breast1
   - lung1
   - lung2
   - test_wavelet_64x64x64
   - test_wavelet_37x37x37

  检查测试案例（由签名为<testCase>_image.nrrd和<testCase>_label.nrrd的图像和掩码文件组成）是否在“dataDirectory”中可用。如果不可用，将从GitHub存储库下载testCase并存储在“dataDirectory”中。如果必要，还会创建“dataDirectory”。如果未指定“dataDirectory”，PyRadiomics将使用临时目录：<TEMPDIR>/pyradiomics/data。

  如果找到或成功下载了测试案例，此函数将返回一个包含两个字符串的元组：
  ``（path/to/image.nrrd，path/to/mask.nrrd）``。在发生错误的情况下返回``（None，None）``。

  .. 注意::
    要获取具有相应单个切片标签的测试案例，将"_2D"附加到testCase。
  """
  global logger, testCases
  label2D = False
  testCase = testCase.lower()
  if testCase.endswith('_2d'):
    label2D = True
    testCase = testCase[:-3]

  if testCase not in testCases:
    raise ValueError('未识别测试用例“%s”！' % testCase)

  logger.debug('获取测试用例 %s', testCase)

  if dataDirectory is None:
    dataDirectory = os.path.join(tempfile.gettempdir(), 'pyradiomics', 'data')
    logger.debug('未指定数据目录，使用临时目录 "%s"', dataDirectory)

  im_name = '%s_image.nrrd' % testCase
  ma_name = '%s_label%s.nrrd' % (testCase, '_2D' if label2D else '')

  def get_or_download(fname):
    target = os.path.join(dataDirectory, fname)
    if os.path.exists(target):
      logger.debug('文件 %s 已下载', fname)
      return target

    # 未找到测试用例文件，因此尝试下载它
    logger.info("本地未找到测试用例文件 %s，正在从GitHub下载...", fname)

    # 首先检查文件夹是否可用
    if not os.path.isdir(dataDirectory):
      logger.debug('创建数据目录：%s', dataDirectory)
      os.makedirs(dataDirectory)

    # 下载测试用例文件（图像和标签）
    url = r'https://github.com/Radiomics/pyradiomics/releases/download/v1.0/%s' % fname

    logger.debug('正在检索位于 %s 的文件', url)
    _, headers = urllib.request.urlretrieve(url, target)

    if headers.get('status', '') == '404 Not Found':
      raise ValueError('无法下载图像文件 %s！' % url)

    logger.info('文件 %s 已下载', fname)
    return target

  logger.debug('获取图像文件')
  imageFile = get_or_download(im_name)

  logger.debug('获取掩码文件')
  maskFile = get_or_download(ma_name)

  return imageFile, maskFile


def getParameterValidationFiles():
  """
  返回参数模式和自定义验证函数的文件位置，这些文件在使用``PyKwalify.core``验证参数文件时需要。
  此函数返回一个包含模式文件的文件位置的元组，作为第一个元素，包含自定义验证函数的python脚本的文件位置作为第二个元素。
  """
  dataDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'schemas'))
  schemaFile = os.path.join(dataDir, 'paramSchema.yaml')
  schemaFuncs = os.path.join(dataDir, 'schemaFuncs.py')
  return schemaFile, schemaFuncs


class _DummyProgressReporter(object):
  """
  此类表示虚拟的进度报告工具，用于实现具有进度报告功能但未启用的情况（当progressReporter未设置或日志级别>INFO时）。

  PyRadiomics期望_getProgressReporter函数返回一个对象，该对象以可迭代对象和“desc”关键字参数作为初始化的第一个位置参数。此外，它应该是可迭代的，其中它迭代在初始化时传递的可迭代对象上，并且应该在“with”语句中使用。

  在这个类中，__iter__函数重定向到初始化时传递的可迭代对象的__iter__函数。__enter__和__exit__函数使其可以在“with”语句中使用。
  """
  def __init__(self, iterable=None, desc='', total=None):
    self.desc = desc  # 不需要描述，但由PyRadiomics提供
    self.iterable = iterable  # 必需的可迭代对象

  def __iter__(self):
    return self.iterable.__iter__()  # 只需在传递的可迭代对象上迭代

  def __enter__(self):
    return self  # __enter__函数应返回自身

  def __exit__(self, exc_type, exc_value, tb):
    pass  # 不需要关闭或处理任何内容，只需指定“pass”

  def update(self, n=1):
    pass  # 不需要更新任何内容，只需指定“pass”


def getProgressReporter(*args, **kwargs):
  """
  此函数返回进度报告工具的实例，如果已设置并且日志级别已定义为级别INFO或DEBUG。在所有其他情况下，将返回虚拟进度报告工具。

  要启用进度报告，应将progressReporter变量设置为类对象（而不是实例），该对象符合以下签名：

  1. 第一个位置参数接受一个可迭代对象，并且有一个关键字参数（'desc'）指定要显示的标签
  2. 可以在“with”语句中使用（即公开一个__enter__和__exit__函数）
  3. 是可迭代的（即至少指定一个__iter__函数，该函数在初始化时迭代在可迭代对象上）。

  还可以创建自己的进度报告工具。为实现此目的，另外指定一个函数`__next__`，并使`__iter__`函数返回“self”。`__next__`函数不接受参数并返回对可迭代对象的__next__函数的调用（即`return self.iterable.__next__()`）。可以在返回语句之前在这个函数中插入任何打印/进度报告调用。
  """
  global handler, progressReporter
  if progressReporter is not None and logging.NOTSET < handler.level <= logging.INFO:
    return progressReporter(*args, **kwargs)
  else:
    return _DummyProgressReporter(*args, **kwargs)

progressReporter = None

# 1. 设置日志记录
debugging = True
logger = logging.getLogger(__name__)  # 'radiomics'
logger.setLevel(logging.INFO)  # 将默认日志记录器级别设置为INFO，以反映大多数情况下的日志文件设置

# 设置一个处理程序以打印到stderr（由setVerbosity()控制）
handler = logging.StreamHandler()
# formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M")  # 另一种格式
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
# 在stderr处理程序中强制级别=WARNING，以防日志记录默认设置不同（问题102）
setVerbosity(logging.WARNING)

# 2. 定义可用的测试用例
testCases = ('brain1', 'brain2', 'breast1', 'lung1', 'lung2', 'test_wavelet_64x64x64', 'test_wavelet_37x37x37')

# 3. 尝试加载和启用C扩展。
cMatrices = None  # 将cMatrices设置为None以防止特征类中出现导入错误。
cShape = None
try:
  from radiomics import _cmatrices as cMatrices  # noqa: F401
  from radiomics import _cshape as cShape  # noqa: F401
except ImportError as e:
  if os.path.isdir(os.path.join(os.path.dirname(__file__), '..', 'data')):
    # 看起来PyRadiomics是从源代码运行的（在这种情况下，必须运行“python setup.py develop”）
    logger.critical('显然是从根目录运行，但无法加载C扩展... '
                    '你是否运行了“python setup.py build_ext --inplace”？')
    raise Exception('显然是从根目录运行，但无法加载C扩展... '
                    '你是否运行了“python setup.py build_ext --inplace”？')
  else:
    logger.critical('加载C扩展时出现错误', exc_info=True)
    raise e

# 4. 枚举在PyRadiomics中实现的特征类和输入图像类型
_featureClasses = None
_imageTypes = None
getFeatureClasses()
getImageTypes()

# 5. 使用版本管理器脚本设置版本
from ._version import get_versions  # noqa: I202

__version__ = get_versions()['version']
del get_versions
