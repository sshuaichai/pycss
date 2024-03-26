# -*- coding: utf-8 -*-
''

from __future__ import print_function

import collections
import glob
import os

import six
'''
数据集层次阅读器
这段代码的主要作用是用于读取和处理数据集的目录结构，并提供了一些功能来操作数据集的文件。
'''
class DatasetHierarchyReader(object):
  def __init__(self, inputDatasetDirectory, filetype='.nrrd'):
    self.inputDatasetDirectory = inputDatasetDirectory
    self.filetype = filetype
    self.DatabaseHierarchyDict = collections.OrderedDict()

  def setInputDatasetDirectory(self, inputDatasetDirectory):
    self.inputDatasetDirectory = inputDatasetDirectory

  def setFiletype(self, filetype):
    self.filetype = filetype

  def ReadDatasetHierarchy(self, create=False): #读取数据集的层次结构，遍历患者目录和研究目录，并将其存储在 DatabaseHierarchyDict 中。
    patientDirectories = glob.glob(os.path.join(self.inputDatasetDirectory, '*'))

    for patientDirectory in patientDirectories:
      self.DatabaseHierarchyDict[patientDirectory] = collections.OrderedDict()
      studyDirectories = glob.glob(os.path.join(patientDirectory, '*'))

      for studyDirectory in studyDirectories:
        self.DatabaseHierarchyDict[patientDirectory][studyDirectory] = collections.OrderedDict()

        subfolders = [dirpath for dirpath in glob.glob(os.path.join(studyDirectory, '*')) if os.path.isdir(dirpath)]

        reconstructionsDirectory, images = self.readReconstructionsDirectory(studyDirectory, subfolders, create=create)
        self.DatabaseHierarchyDict[patientDirectory][studyDirectory]["reconstructions"] = images

        resourcesDirectory, resources = self.readResourcesDirectory(studyDirectory, subfolders, create=create)
        self.DatabaseHierarchyDict[patientDirectory][studyDirectory]["resources"] = resources

        segmentationsDirectory, labels = self.readSegmentationsDirectory(studyDirectory, subfolders, create=create)
        self.DatabaseHierarchyDict[patientDirectory][studyDirectory]["segmentations"] = labels

    return self.DatabaseHierarchyDict

  def readReconstructionsDirectory(self, studyDirectory, subfolders, create=False): #读取重建目录，并返回该目录以及其中的图像文件列表。
    images = []
    recDirectory = "NONE"
    try:
      recDirectory = [item for item in subfolders if 'reconstructions' in os.path.basename(item).lower()][0]
      images = [item for item in glob.glob(os.path.join(recDirectory, "*")) if self.filetype in os.path.basename(item)]
    except IndexError:
      if create:
        recDirectory = os.path.join(studyDirectory, "Reconstructions")
        if not os.path.exists(recDirectory):
          os.mkdir(recDirectory)
          print("\tCreated:", recDirectory)

    return recDirectory, images

  def readSegmentationsDirectory(self, studyDirectory, subfolders, create=False): #读取分割目录，并返回该目录以及其中的标签文件列表。
    labels = []
    segDirectory = "NONE"
    try:
      segDirectory = [item for item in subfolders if 'segmentations' in os.path.basename(item).lower()][0]
      labels = [item for item in glob.glob(os.path.join(segDirectory, "*")) if self.filetype in os.path.basename(item)]
    except IndexError:
      if create:
        segDirectory = os.path.join(studyDirectory, "Segmentations")
        if not os.path.exists(segDirectory):
          os.mkdir(segDirectory)
          print("\tCreated:", segDirectory)

    return segDirectory, labels

  def readResourcesDirectory(self, studyDirectory, subfolders, create=False):#读取资源目录，并返回该目录以及其中的资源文件列表。
    resources = []
    resDirectory = "NONE"
    try:
      resDirectory = [item for item in subfolders if 'resources' in os.path.basename(item).lower()][0]
      resources = [item for item in glob.glob(os.path.join(resDirectory, "*"))]
    except IndexError:
      if create:
        resDirectory = os.path.join(studyDirectory, "Resources")
        if not os.path.exists(resDirectory):
          os.mkdir(resDirectory)
          print("\tCreated:", resDirectory)

    return resDirectory, resources

  def findImageAndLabelPair(self, imageFilepaths, maskFilepaths, keywordSettings): #查找符合关键字条件的图像文件和标签文件对。
    """
  接受图像文件路径列表、掩码/标签文件路径列表以及关键字设置的字典，格式如下：

    keywordSettings['image'] = ""
    keywordSettings['imageExclusion'] = ""
    keywordSettings['mask'] = ""
    keywordSettings['maskExclusion'] = ""

   其中每个字段都是一串用逗号分隔的单词（大小写和空格无关紧要）。
   输出是满足关键字条件的图像文件路径和掩码/标签文件路径对。
    """

    keywordSettings = {k: [str(keyword.strip()) for keyword in v.split(',')]
                       for (k, v) in six.iteritems(keywordSettings)}

    matchedImages = []
    for imageFilepath in imageFilepaths:
      imageFilename = str(os.path.basename(imageFilepath))
      if self.testString(imageFilename, keywordSettings['image'], keywordSettings['imageExclusion']):
        matchedImages.append(imageFilepath)

    matchedMasks = []
    for maskFilepath in maskFilepaths:
      maskFilename = str(os.path.basename(maskFilepath))
      if self.testString(maskFilename, keywordSettings['mask'], keywordSettings['maskExclusion']):
        matchedMasks.append(maskFilepath)

    if len(matchedImages) < 1:
      print("ERROR: No Images Matched")
    elif len(matchedImages) > 1:
      print("ERROR: Multiple Images Matched")

    if len(matchedMasks) < 1:
      print("ERROR: No Masks Matched")
    elif len(matchedMasks) > 1:
      print("ERROR: Multiple Masks Matched")

    if (len(matchedImages) == 1) and (len(matchedMasks) == 1):
      return matchedImages[0], matchedMasks[0]
    else:
      return None, None

  def testString(self, fileName, inclusionKeywords, exclusionKeywords): #测试字符串是否满足包含和排除关键字的条件。
    fileName = fileName.upper()
    inclusionKeywords = [keyword.upper() for keyword in inclusionKeywords if (keyword != '')]
    exclusionKeywords = [keyword.upper() for keyword in exclusionKeywords if (keyword != '')]

    result = False
    if (len(inclusionKeywords) == 0) and (len(exclusionKeywords) > 0):
      if (not any(keyword in fileName for keyword in exclusionKeywords)):
        result = True
    elif (len(inclusionKeywords) > 0) and (len(exclusionKeywords) == 0):
      if (all(keyword in fileName for keyword in inclusionKeywords)):
        result = True
    elif (len(inclusionKeywords) > 0) and (len(exclusionKeywords) > 0):
      if (all(keyword in fileName for keyword in inclusionKeywords)) and \
        (not any(keyword in fileName for keyword in exclusionKeywords)):
        result = True
    elif (len(inclusionKeywords) == 0) and (len(exclusionKeywords) == 0):
      result = True

    return result





