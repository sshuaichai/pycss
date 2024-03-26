# 为了方便，将collections和numpy导入到"pyradiomics"命名空间中
import collections  # noqa: F401  # 导入collections库，用于支持容器数据类型
import inspect  # 导入inspect库，用于收集模块和类的信息
import logging  # 导入logging库，用于记录日志
import os  # 导入os库，用于处理文件和目录
import pkgutil  # 导入pkgutil库，用于处理包和模块
import sys  # 导入sys库，用于访问与Python解释器紧密相关的变量和函数
import tempfile  # 导入tempfile库，用于创建临时文件和目录

import numpy  # noqa: F401  # 导入numpy库，用于数值计算
from six.moves import urllib  # 从six库导入urllib模块，用于兼容Python 2和3的URL处理

from SKMRradiomics import imageoperations  # 从SKMRradiomics包导入imageoperations模块

def deprecated(func):
  """
  用作装饰器的函数，标记函数为已弃用。这用于确保在启用'所有'特征时，不会将已弃用的特征函数添加到启用特征列表中。
  """
  func._is_deprecated = True
  return func

def setVerbosity(level):
  """
  更改PyRadiomics在提取过程中应打印出的信息量。级别越低，打印到输出(stderr)的信息就越多。

  使用``level``参数（Python定义的日志级别），可以设置以下级别：

  - 60: 安静模式，不打印任何消息
  - 50: 只打印"CRITICAL"级别的日志消息
  - 40: 打印"ERROR"及以上级别的日志消息
  - 30: 打印"WARNING"及以上级别的日志消息
  - 20: 打印"INFO"及以上级别的日志消息
  - 10: 打印"DEBUG"及以上级别的日志消息（即所有日志消息）

  默认情况下，radiomics记录器设置为"INFO"级别，stderr处理器设置为"WARNING"级别。因此，通过向radiomics记录器添加适当的处理器，可以轻松设置存储"INFO"及以上级别提取日志消息的日志，而stderr的输出仍然只包含警告和错误。

  .. note::

    此函数假设在工具箱初始化时添加到radiomics记录器的处理器没有从记录器处理器中移除，因此仍然是第一个处理器。

  .. note::

    这不影响记录器本身的级别（例如，如果详细级别=3，DEBUG级别的日志消息仍然可以存储在日志文件中，如果向记录器添加了适当的处理器并将记录器的日志级别设置为正确的级别。*例外：如果详细级别设置为DEBUG，记录器的级别也降低到DEBUG。如果然后再次提高详细级别，记录器级别将保持DEBUG。*
  """
  global logger, handler
  if level < 10:  # 最低级别：DEBUG
    level = 10
  if level > 60:  # 最高级别=50（CRITICAL），级别60导致“安静”模式
    level = 60

  handler.setLevel(level)
  if handler.level < logger.level:  # 如有必要，降低记录器的级别
    logger.setLevel(level)

def getFeatureClasses():
  """
  使用pkgutil遍历radiomics包的所有模块，并随后导入这些模块。

  返回一个包含所有包含featureClasses的模块的字典，键为模块名，值为featureClass的抽象类对象。假设每个模块只有一个featureClass。

  通过inspect.getmembers实现。如果模块包含一个类成员，其名称以'Radiomics'开头并且继承自:py:class:`radiomics.base.RadiomicsFeaturesBase`，则添加模块。

  此迭代仅在工具箱初始化时运行一次（第一次调用时），后续调用返回第一次调用创建的字典。
  """
  global _featureClasses
  if _featureClasses is None:  # 在第一次调用时，枚举可能的特征类并导入PyRadiomics模块
    _featureClasses = {}
    for _, mod, _ in pkgutil.iter_modules([os.path.dirname(__file__)]):
      if str(mod).startswith('_'):  # 跳过加载'私有'类，这些类不包含特征类
        continue
      __import__('SKMRradiomics.' + mod)
      module = sys.modules['SKMRradiomics.' + mod]
      attributes = inspect.getmembers(module, inspect.isclass)
      for a in attributes:
        if a[0].startswith('Radiomics'):
          for parentClass in inspect.getmro(a[1])[1:]:  # 只包括继承自RadiomicsFeaturesBase的类
            if parentClass.__name__ == 'RadiomicsFeaturesBase':
              _featureClasses[mod] = a[1]
              break

  return _featureClasses

def getImageTypes():
  """
  返回可能的图像类型列表（即可能的过滤器和“原始”未过滤的图像类型）。此函数通过将签名("get<imageType>Image")与在:ref:`imageoperations <radiomics-imageoperations-label>`中定义的函数匹配，动态地找到图像类型。返回包含可用图像类型名称的列表（对应函数名称的<imageType>部分）。

  此迭代仅在工具箱初始化时发生一次。在后续调用中，找到的结果被存储并返回。
  """
  global _imageTypes
  if _imageTypes is None:  # 在第一次调用时，枚举可能的输入图像类型（原始和任何过滤器）
    _imageTypes = [member[3:-5] for member in dir(imageoperations)
                   if member.startswith('get') and member.endswith("Image")]

  return _imageTypes

def getTestCase(testCase, dataDirectory=None):
  """
  此函数为测试PyRadiomics提供图像和掩码。可以选择七个测试案例之一：

   - brain1
   - brain2
   - breast1
   - lung1
   - lung2
   - test_wavelet_64x64x64
   - test_wavelet_37x37x37

  检查测试案例（由具有签名<testCase>_image.nrrd和<testCase>_label.nrrd的图像和掩码文件组成）是否在``dataDirectory``中可用。如果不可用，测试案例将从GitHub仓库下载并存储在``dataDirectory``中。如果需要，还会创建``dataDirectory``。如果没有指定``dataDirectory``，PyRadiomics将使用临时目录：<TEMPDIR>/pyradiomics/data。

  如果测试案例已成功找到或下载，此函数返回两个字符串的元组：``(path/to/image.nrrd, path/to/mask.nrrd)``。在出现错误的情况下返回``(None, None)``。
  """
  global logger, testCases
  if testCase not in testCases:
    logger.error('测试案例 "%s" 未被识别!', testCase)
    return None, None

  logger.debug('获取测试案例 %s', testCase)

  if dataDirectory is None:
    dataDirectory = os.path.join(tempfile.gettempdir(), 'pyradiomics', 'data')
    logger.debug('未指定数据目录，使用临时目录 "%s"', dataDirectory)

  # 检查测试案例是否已经被下载。
  imageFile = os.path.join(dataDirectory, '%s_image.nrrd' % testCase)
  maskFile = os.path.join(dataDirectory, '%s_label.nrrd' % testCase)
  if os.path.isfile(imageFile) and os.path.isfile(maskFile):
    logger.info('测试案例已下载')
    return imageFile, maskFile

  # 未找到测试案例，尝试下载
  logger.info("本地未找到测试案例 %s，正在下载测试案例...", testCase)

  try:
    # 首先检查文件夹是否可用
    if not os.path.isdir(dataDirectory):
      logger.debug('创建数据目录: %s', dataDirectory)
      os.makedirs(dataDirectory)

    # 下载测试案例文件（图像和标签）
    url = r'https://github.com/Radiomics/pyradiomics/releases/download/v1.0/%s_%s.nrrd'

    logger.debug('正在检索图像 %s', url % (testCase, 'image'))
    fname, headers = urllib.request.urlretrieve(url % (testCase, 'image'), imageFile)
    if headers.get('status', '') == '404 Not Found':
      logger.warning('无法在 %s 下载图像文件!', url % (testCase, 'image'))
      return None, None

    logger.debug('正在检索掩码 %s', url % (testCase, 'label'))
    fname, headers = urllib.request.urlretrieve(url % (testCase, 'label'), maskFile)
    if headers.get('status', '') == '404 Not Found':
      logger.warning('无法在 %s 下载掩码文件!', url % (testCase, 'label'))

    logger.info('测试案例 %s 已下载', testCase)
    return imageFile, maskFile
  except Exception:
    logger.error('下载失败!', exc_info=True)
    return None, None

def getParameterValidationFiles():
  """
  返回参数模式和自定义验证函数的文件位置，这些文件在使用``PyKwalify.core``验证参数文件时需要。
  此函数返回一个元组，其中模式文件的位置为第一个元素，包含自定义验证函数的Python脚本为第二个元素。
  """
  dataDir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'schemas'))
  schemaFile = os.path.join(dataDir, 'paramSchema.yaml')
  schemaFuncs = os.path.join(dataDir, 'schemaFuncs.py')
  return schemaFile, schemaFuncs

class _DummyProgressReporter(object):
  """
  这个类代表虚拟进度报告器，用于实现了进度报告的地方，但未启用（当progressReporter未设置或详细级别>INFO时）。

  PyRadiomics期望_getProgressReporter函数返回一个对象，该对象在初始化时接受一个iterable和'desc'关键字参数。此外，它应该是可迭代的，其中它迭代在初始化时传递的iterable，它应该在'with'语句中使用。

  在这个类中，__iter__函数重定向到初始化时传递的iterable的__iter__函数。__enter__和__exit__函数使其能够在'with'语句中使用。
  """
  def __init__(self, iterable, desc=''):
    self.desc = desc  # 描述不是必需的，但是PyRadiomics提供了
    self.iterable = iterable  # 需要iterable

  def __iter__(self):
    return self.iterable.__iter__()  # 只是在初始化时迭代传递的iterable

  def __enter__(self):
    return self  # __enter__函数应返回自身

  def __exit__(self, exc_type, exc_value, tb):
    pass  # 无需关闭或处理任何东西，所以只需指定'pass'

def getProgressReporter(*args, **kwargs):
  """
  如果设置了progressReporter并且日志级别定义在INFO或DEBUG级别，则此函数返回progressReporter的实例。在所有其他情况下，返回一个虚拟进度报告器。

  要启用进度报告，progressReporter变量应该设置为一个类对象（NOT实例），它符合以下签名：

  1. 接受一个iterable作为第一个位置参数和一个关键字参数('desc')，指定要显示的标签
  2. 可以在'with'语句中使用（即暴露一个__enter__和__exit__函数）
  3. 是可迭代的（即至少指定一个__iter__函数，它迭代在初始化时传递的iterable）

  也可以创建自己的进度报告器。为此，另外指定一个函数`__next__`，并让`__iter__`函数返回`self`。`__next__`函数不接受任何参数并返回iterable的`__next__`函数调用（即`return self.iterable.__next__()`）。然后可以在此函数返回语句之前插入任何打印/进度报告调用。
  """
  global handler, progressReporter
  if progressReporter is not None and logging.NOTSET < handler.level <= logging.INFO:
    return progressReporter(*args, **kwargs)
  else:
    return _DummyProgressReporter(*args, **kwargs)

progressReporter = None

# 1. 设置日志
debugging = True
logger = logging.getLogger(__name__)  # 'radiomics' 日志记录器
logger.setLevel(logging.INFO)  # 将记录器的默认级别设置为INFO，以反映日志文件最常见的设置

# 设置一个处理器来打印到stderr（由setVerbosity()控制）
handler = logging.StreamHandler()
# formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", "%Y-%m-%d %H:%M")  # 另一种格式
formatter = logging.Formatter("%(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
# 强制级别=WARNING的stderr处理器，以防日志默认设置不同（问题102）
setVerbosity(logging.WARNING)

# 2. 定义可用的测试案例
testCases = ('brain1', 'brain2', 'breast1', 'lung1', 'lung2', 'test_wavelet_64x64x64', 'test_wavelet_37x37x37')

# 3. 尝试加载并启用C扩展。
cMatrices = None  # 将cMatrices设置为None，以防止在特征类中导入错误。
cShape = None
try:
  from . import _cmatrices as cMatrices  # noqa: F401
  from . import _cshape as cShape  # noqa: F401
except ImportError as e:
  if os.path.isdir(os.path.join(os.path.dirname(__file__), '..', 'data')):
    # 看起来像是从源代码运行PyRadiomics（在这种情况下，必须已经运行了"setup.py develop"）
    logger.critical('看起来像是从根目录运行，但无法加载C扩展... '
                    '你运行了"python setup.py build_ext --inplace"吗？')
    raise Exception('看起来像是从根目录运行，但无法加载C扩展... '
                    '你运行了"python setup.py build_ext --inplace"吗？')
  else:
    logger.critical('加载C扩展失败', exc_info=True)
    raise e

# 4. 枚举PyRadiomics中实现的特征类和输入图像类型
_featureClasses = None
_imageTypes = None
getFeatureClasses()
getImageTypes()

# 5. 使用versioneer脚本设置版本
from ._version import get_versions  # noqa: I202

__version__ = get_versions()['version']
del get_versions
