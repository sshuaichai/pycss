# 该文件帮助计算从 git-archive tarball（例如那些通过 github 的 download-from-tag
# 功能提供的）获取的源代码树中的版本号。通过 setup.py sdist 构建的发行版 tarball
# 和通过 setup.py build 生成的构建目录将包含一个更短的文件，该文件仅包含计算出的版本号。

# 该文件被释放到公共领域。由 versioneer-0.17 生成（https://github.com/warner/python-versioneer）

"""Git 实现的 _version.py。"""

from __future__ import print_function  # 从 __future__ 导入 print 函数，以便在 Python 2.x 中使用 Python 3.x 的 print 函数

import errno  # 导入 errno 模块，用于定义标准的 errno 系统符号
import os  # 导入 os 模块，用于与操作系统交互
import re  # 导入 re 模块，用于正则表达式匹配
import subprocess  # 导入 subprocess 模块，用于启动新进程并与之通信
import sys  # 导入 sys 模块，用于访问与 Python 解释器紧密相关的变量和函数


def get_keywords():
  """获取查找版本信息所需的关键字。"""
  # 这些字符串将被 git 在 git-archive 过程中替换。
  # setup.py/versioneer.py 将会 grep 这些变量名，因此它们必须
  # 每个定义在自己的行上。_version.py 将仅仅调用 get_keywords()。
  git_refnames = "$Format:%d$"  # git 引用名格式
  git_full = "$Format:%H$"  # git 完整哈希格式
  git_date = "$Format:%ci$"  # git 日期格式
  keywords = {"refnames": git_refnames, "full": git_full, "date": git_date}  # 将关键字打包成字典
  return keywords  # 返回关键字字典


class VersioneerConfig:
  """用于存储 Versioneer 配置参数的容器。"""


def get_config():
  """创建、填充并返回 VersioneerConfig() 对象。"""
  # 当 'setup.py versioneer' 创建 _version.py 时，这些字符串会被填充
  cfg = VersioneerConfig()  # 创建 VersioneerConfig 实例
  cfg.VCS = "git"  # 版本控制系统为 git
  cfg.style = "pep440-post"  # 版本号风格
  cfg.tag_prefix = ""  # 标签前缀
  cfg.parentdir_prefix = "None"  # 父目录前缀
  cfg.versionfile_source = "radiomics/_version.py"  # 版本文件源路径
  cfg.verbose = False  # 是否启用详细模式
  return cfg  # 返回配置对象


class NotThisMethod(Exception):
  """如果某方法不适用于当前场景，则抛出此异常。"""


LONG_VERSION_PY = {}  # 长版本 Python 字典
HANDLERS = {}  # 处理器字典


def register_vcs_handler(vcs, method):  # 装饰器
  """将方法标记为特定 VCS 的处理器的装饰器。"""

  def decorate(f):
    """将 f 存储在 HANDLERS[vcs][method] 中。"""
    if vcs not in HANDLERS:
      HANDLERS[vcs] = {}  # 如果 vcs 不在 HANDLERS 中，则添加
    HANDLERS[vcs][method] = f  # 将函数 f 设置为对应的处理器
    return f  # 返回函数 f

  return decorate  # 返回装饰器函数


def run_command(commands, args, cwd=None, verbose=False, hide_stderr=False,
                env=None):
  """调用给定的命令。"""
  assert isinstance(commands, list)  # 断言 commands 是列表
  p = None  # 初始化进程变量
  for c in commands:
    try:
      dispcmd = str([c] + args)  # 显示命令
      # 记住 shell=False，所以在 Windows 上使用 git.cmd 而不仅仅是 git
      p = subprocess.Popen([c] + args, cwd=cwd, env=env,
                           stdout=subprocess.PIPE,
                           stderr=(subprocess.PIPE if hide_stderr
                                   else None))  # 启动子进程
      break  # 成功启动进程后跳出循环
    except EnvironmentError:
      e = sys.exc_info()[1]  # 捕获异常信息
      if e.errno == errno.ENOENT:
        continue  # 如果是文件不存在的错误，尝试下一个命令
      if verbose:
        print("unable to run %s" % dispcmd)  # 打印错误信息
        print(e)
      return None, None  # 返回 None 表示命令执行失败
  else:
    if verbose:
      print("unable to find command, tried %s" % (commands,))  # 如果所有命令都尝试失败，打印错误信息
    return None, None  # 返回 None 表示找不到命令
  stdout = p.communicate()[0].strip()  # 获取标准输出并去除尾部空格
  if sys.version_info[0] >= 3:
    stdout = stdout.decode()  # 如果是 Python 3，则解码输出
  if p.returncode != 0:
    if verbose:
      print("unable to run %s (error)" % dispcmd)  # 如果进程返回非零值，打印错误信息
      print("stdout was %s" % stdout)
    return None, p.returncode  # 返回 None 和进程返回码
  return stdout, p.returncode  # 返回标准输出和进程返回码


# 以下是一系列函数和类，用于从 git 描述符、父目录名等获取版本信息，
# 并根据不同的风格（如 PEP 440）渲染版本字符串。
# 这些函数涵盖了从 git 仓库中提取版本信息的多种方法，包括直接从 git 标签、
# 关键字替换以及文件和目录命名约定中提取。
# 它们支持灵活的版本号生成策略，适用于不同的开发和发布流程。

def versions_from_parentdir(parentdir_prefix, root, verbose):
  """尝试从父目录名称确定版本。

  源代码 tarball 通常解压到一个包含项目名称和版本字符串的目录中。我们还将支持向上搜索
  两级目录以查找适当命名的父目录"""
  rootdirs = []

  for i in range(3):
    dirname = os.path.basename(root)
    if dirname.startswith(parentdir_prefix):
      return {"version": dirname[len(parentdir_prefix):],
              "full-revisionid": None,
              "dirty": False, "error": None, "date": None}
    else:
      rootdirs.append(root)
      root = os.path.dirname(root)  # 上一级目录

  if verbose:
    print("Tried directories %s but none started with prefix %s" %
          (str(rootdirs), parentdir_prefix))
  raise NotThisMethod("rootdir doesn't start with parentdir_prefix")


@register_vcs_handler("git", "get_keywords")
def git_get_keywords(versionfile_abs):
  """从给定文件中提取版本信息。"""
  # 嵌入在 _version.py 中的代码可以直接获取这些关键字的值。当从 setup.py 使用时，
  # 我们不想导入 _version.py，所以我们使用正则表达式代替。这个函数不会从 _version.py 调用。
  keywords = {}
  try:
    f = open(versionfile_abs, "r")
    for line in f.readlines():
      if line.strip().startswith("git_refnames ="):
        mo = re.search(r'=\s*"(.*)"', line)
        if mo:
          keywords["refnames"] = mo.group(1)
      if line.strip().startswith("git_full ="):
        mo = re.search(r'=\s*"(.*)"', line)
        if mo:
          keywords["full"] = mo.group(1)
      if line.strip().startswith("git_date ="):
        mo = re.search(r'=\s*"(.*)"', line)
        if mo:
          keywords["date"] = mo.group(1)
    f.close()
  except EnvironmentError:
    pass
  return keywords


@register_vcs_handler("git", "keywords")
def git_versions_from_keywords(keywords, tag_prefix, verbose):
  """从 git 关键字获取版本信息。"""
  if not keywords:
    raise NotThisMethod("没有关键字，很奇怪")
  date = keywords.get("date")
  if date is not None:
    # git-2.2.0 添加了 "%cI"，它扩展为符合 ISO-8601 的日期戳。
    # 然而我们更喜欢 "%ci"（它扩展为一个“类 ISO-8601”字符串，我们必须随后编辑它以使其符合标准），
    # 因为它自 git-1.5.3 以来就存在了，发现我们正在使用哪个版本或使用旧版本太困难了。
    date = date.strip().replace(" ", "T", 1).replace(" ", "", 1)
  refnames = keywords["refnames"].strip()
  if refnames.startswith("$Format"):
    if verbose:
      print("关键字未展开，不使用")
    raise NotThisMethod("关键字未展开，不是 git-archive tarball")
  refs = set([r.strip() for r in refnames.strip("()").split(",")])
  # 从 git-1.8.3 开始，标签被列为 "tag: foo-1.0" 而不仅仅是 "foo-1.0"。
  # 如果我们看到一个 "tag: " 前缀，优先选择这些。
  TAG = "tag: "
  tags = set([r[len(TAG):] for r in refs if r.startswith(TAG)])
  if not tags:
    # 要么我们使用的是 git < 1.8.3，要么真的没有标签。我们使用一个启发式方法：
    # 假设所有版本标签都有数字。旧的 git %d 扩展行为类似于 git log --decorate=short
    # 并去除了能让我们区分分支和标签的 refs/heads/ 和 refs/tags/ 前缀。
    # 通过忽略没有数字的 refnames，我们过滤掉许多常见的分支名称，如 "release" 和
    # "stabilization"，以及 "HEAD" 和 "master"。
    tags = set([r for r in refs if re.search(r'\d', r)])
    if verbose:
      print("丢弃 '%s'，没有数字" % ",".join(refs - tags))
  if verbose:
    print("可能的标签: %s" % ",".join(sorted(tags)))
  for ref in sorted(tags):
    # 排序将优先选择例如 "2.0" 而不是 "2.0rc1"
    if ref.startswith(tag_prefix):
      r = ref[len(tag_prefix):]
      if verbose:
        print("选择 %s" % r)
      return {"version": r,
              "full-revisionid": keywords["full"].strip(),
              "dirty": False, "error": None,
              "date": date}
  # 没有合适的标签，所以版本是 "0+unknown"，但完整的 hex 仍然存在
  if verbose:
    print("没有合适的标签，使用未知 + 完整的修订 id")
  return {"version": "0+unknown",
          "full-revisionid": keywords["full"].strip(),
          "dirty": False, "error": "没有合适的标签", "date": None}


@register_vcs_handler("git", "pieces_from_vcs")
def git_pieces_from_vcs(tag_prefix, root, verbose, run_command=run_command):
  """从源代码树的根目录中使用 'git describe' 获取版本。

  仅当 git-archive 'subst' 关键字*未*展开，且 _version.py 尚未被重写为短版本字符串时调用，
  意味着我们处于检出的源代码树内。"""
  GITS = ["git"]
  if sys.platform == "win32":
    GITS = ["git.cmd", "git.exe"]

  out, rc = run_command(GITS, ["rev-parse", "--git-dir"], cwd=root, hide_stderr=True)
  if rc != 0:
    if verbose:
      print("目录 %s 不在 git 控制之下" % root)
    raise NotThisMethod("'git rev-parse --git-dir' 返回错误")

  # 如果存在与 tag_prefix 匹配的标签，这将产生 TAG-NUM-gHEX[-dirty]
  # 如果没有，则产生 HEX[-dirty]（没有 NUM）
  describe_out, rc = run_command(GITS, ["describe", "--tags", "--dirty",
                                        "--always", "--long",
                                        "--match", "%s*" % tag_prefix], cwd=root)
  # --long 在 git-1.5.5 中被添加
  if describe_out is None:
    raise NotThisMethod("'git describe' 失败")
  describe_out = describe_out.strip()
  full_out, rc = run_command(GITS, ["rev-parse", "HEAD"], cwd=root)
  if full_out is None:
    raise NotThisMethod("'git rev-parse' 失败")
  full_out = full_out.strip()

  pieces = {}
  pieces["long"] = full_out
  pieces["short"] = full_out[:7]  # 或许之后会改进
  pieces["error"] = None

  # 解析 describe_out。它将像 TAG-NUM-gHEX[-dirty] 或 HEX[-dirty]
  # TAG 可能包含连字符。
  git_describe = describe_out

  # 寻找 -dirty 后缀
  dirty = git_describe.endswith("-dirty")
  pieces["dirty"] = dirty
  if dirty:
    git_describe = git_describe[:git_describe.rindex("-dirty")]

  # 现在我们有 TAG-NUM-gHEX 或 HEX

  if "-" in git_describe:
    # TAG-NUM-gHEX
    mo = re.search(r'^(.+)-(\d+)-g([0-9a-f]+)$', git_describe)
    if not mo:
      # 无法解析。也许 git-describe 表现异常？
      pieces["error"] = ("无法解析 git-describe 输出: '%s'" % describe_out)
      return pieces

    # 标签
    full_tag = mo.group(1)
    if not full_tag.startswith(tag_prefix):
      if verbose:
        fmt = "标签 '%s' 不以前缀 '%s' 开始"
        print(fmt % (full_tag, tag_prefix))
      pieces["error"] = ("标签 '%s' 不以前缀 '%s' 开始" % (full_tag, tag_prefix))
      return pieces
    pieces["closest-tag"] = full_tag[len(tag_prefix):]

    # 距离：自标签以来的提交数
    pieces["distance"] = int(mo.group(2))

    # 提交：短十六进制修订 ID
    pieces["short"] = mo.group(3)

  else:
    # HEX：没有标签
    pieces["closest-tag"] = None
    count_out, rc = run_command(GITS, ["rev-list", "HEAD", "--count"], cwd=root)
    pieces["distance"] = int(count_out)  # 提交总数

  # 提交日期：参见 git_versions_from_keywords() 中的 ISO-8601 注释
  date = run_command(GITS, ["show", "-s", "--format=%ci", "HEAD"], cwd=root)[0].strip()
  pieces["date"] = date.strip().replace(" ", "T", 1).replace(" ", "", 1)

  return pieces


def plus_or_dot(pieces):
  """如果我们还没有加号，则返回加号，否则返回点号。"""
  if "+" in pieces.get("closest-tag", ""):
    return "."
  return "+"


def render_pep440(pieces):
  """构建版本字符串，带有发布后的“本地版本标识符”。

  我们的目标：TAG[+DISTANCE.gHEX[.dirty]]。注意，如果你得到一个标记的构建然后修改它，
  你将得到 TAG+0.gHEX.dirty

  异常：
  1：没有标签。git_describe 只是 HEX。0+untagged.DISTANCE.gHEX[.dirty]"""
  if pieces["closest-tag"]:
    rendered = pieces["closest-tag"]
    if pieces["distance"] or pieces["dirty"]:
      rendered += plus_or_dot(pieces)
      rendered += "%d.g%s" % (pieces["distance"], pieces["short"])
      if pieces["dirty"]:
        rendered += ".dirty"
  else:
    # 异常 #1
    rendered = "0+untagged.%d.g%s" % (pieces["distance"], pieces["short"])
    if pieces["dirty"]:
      rendered += ".dirty"
  return rendered


def render_pep440_pre(pieces):
  """TAG[.post.devDISTANCE] -- 没有 -dirty。

  异常：
  1：没有标签。0.post.devDISTANCE"""
  if pieces["closest-tag"]:
    rendered = pieces["closest-tag"]
    if pieces["distance"]:
      rendered += ".post.dev%d" % pieces["distance"]
  else:
    # 异常 #1
    rendered = "0.post.dev%d" % pieces["distance"]
  return rendered


def render_pep440_post(pieces):
  """TAG[.postDISTANCE[.dev0]+gHEX]。

  ".dev0" 表示脏。注意 .dev0 会倒序排序
  （一个脏的树会看起来“更旧”比相应的干净的树），
  但无论如何你不应该发布带有 -dirty 的软件。

  异常：
  1：没有标签。0.postDISTANCE[.dev0]"""
  if pieces["closest-tag"]:
    rendered = pieces["closest-tag"]
    if pieces["distance"] or pieces["dirty"]:
      rendered += ".post%d" % pieces["distance"]
      if pieces["dirty"]:
        rendered += ".dev0"
      rendered += plus_or_dot(pieces)
      rendered += "g%s" % pieces["short"]
  else:
    # 异常 #1
    rendered = "0.post%d" % pieces["distance"]
    if pieces["dirty"]:
      rendered += ".dev0"
    rendered += "+g%s" % pieces["short"]
  return rendered


def render_pep440_old(pieces):
  """TAG[.postDISTANCE[.dev0]]。

  ".dev0" 表示脏。

  异常：
  1：没有标签。0.postDISTANCE[.dev0]"""
  if pieces["closest-tag"]:
    rendered = pieces["closest-tag"]
    if pieces["distance"] or pieces["dirty"]:
      rendered += ".post%d" % pieces["distance"]
      if pieces["dirty"]:
        rendered += ".dev0"
  else:
    # 异常 #1
    rendered = "0.post%d" % pieces["distance"]
    if pieces["dirty"]:
      rendered += ".dev0"
  return rendered


def render_git_describe(pieces):
  """TAG[-DISTANCE-gHEX][-dirty]。

  类似于 'git describe --tags --dirty --always'。

  异常：
  1：没有标签。HEX[-dirty]（注意：没有 'g' 前缀）"""
  if pieces["closest-tag"]:
    rendered = pieces["closest-tag"]
    if pieces["distance"]:
      rendered += "-%d-g%s" % (pieces["distance"], pieces["short"])
  else:
    # 异常 #1
    rendered = pieces["short"]
  if pieces["dirty"]:
    rendered += "-dirty"
  return rendered


def render_git_describe_long(pieces):
  """TAG-DISTANCE-gHEX[-dirty]。

  类似于 'git describe --tags --dirty --always -long'。
  距离/哈希是无条件的。

  异常：
  1：没有标签。HEX[-dirty]（注意：没有 'g' 前缀）"""
  if pieces["closest-tag"]:
    rendered = pieces["closest-tag"]
    rendered += "-%d-g%s" % (pieces["distance"], pieces["short"])
  else:
    # 异常 #1
    rendered = pieces["short"]
  if pieces["dirty"]:
    rendered += "-dirty"
  return rendered


def render(pieces, style):
  """将给定的版本片段渲染成请求的风格。"""
  if pieces["error"]:
    return {"version": "unknown",  # 如果存在错误，版本未知
            "full-revisionid": pieces.get("long"),
            "dirty": None,
            "error": pieces["error"],
            "date": None}

  if not style or style == "default":
    style = "pep440"  # 默认风格

  if style == "pep440":
    rendered = render_pep440(pieces)
  elif style == "pep440-pre":
    rendered = render_pep440_pre(pieces)
  elif style == "pep440-post":
    rendered = render_pep440_post(pieces)
  elif style == "pep440-old":
    rendered = render_pep440_old(pieces)
  elif style == "git-describe":
    rendered = render_git_describe(pieces)
  elif style == "git-describe-long":
    rendered = render_git_describe_long(pieces)
  else:
    raise ValueError("未知的风格 '%s'" % style)  # 如果风格未知，抛出异常

  return {"version": rendered, "full-revisionid": pieces["long"],
          "dirty": pieces["dirty"], "error": None,
          "date": pieces.get("date")}  # 返回渲染后的版本信息


def get_versions():
  """获取版本信息，或在无法做到时返回默认值。"""
  # 我在 _version.py 中，它位于 ROOT/VERSIONFILE_SOURCE。如果我们有
  # __file__，我们可以从那里向后工作到根目录。一些
  # py2exe/bbfreeze/非CPython实现不处理 __file__，在这种情况下
  # 我们只能使用扩展的关键字。

  cfg = get_config()
  verbose = cfg.verbose

  try:
    return git_versions_from_keywords(get_keywords(), cfg.tag_prefix, verbose)
  except NotThisMethod:
    pass  # 如果方法不适用，尝试下一个方法

  try:
    root = os.path.realpath(__file__)
    # versionfile_source 是从源代码树顶部（可能存在 .git 目录的地方）
    # 到这个文件的相对路径。反转这个来从 __file__ 找到根目录。
    for i in cfg.versionfile_source.split('/'):
      root = os.path.dirname(root)
  except NameError:
    return {"version": "0+unknown", "full-revisionid": None,
            "dirty": None, "error": "无法找到源代码树的根目录",
            "date": None}

  try:
    pieces = git_pieces_from_vcs(cfg.tag_prefix, root, verbose)
    return render(pieces, cfg.style)
  except NotThisMethod:
    pass  # 如果方法不适用，尝试下一个方法

  try:
    if cfg.parentdir_prefix:
      return versions_from_parentdir(cfg.parentdir_prefix, root, verbose)
  except NotThisMethod:
    pass  # 如果方法不适用，尝试下一个方法

  return {"version": "0+unknown", "full-revisionid": None,
          "dirty": None, "error": "无法计算版本", "date": None}  # 如果所有方法都不适用，返回默认值










