import collections
import csv
import logging
import os
import yaml

import SimpleITK as sitk
import radiomics
from radiomics import featureextractor

'''
定义参数进行批处理：
输入CSV文件Task02_Heart.csv
使用指定参数Params.yaml
进度日志文件pyrad_log.txt
输出CSV文件radiomics_Primitive.csv：进计算原始特征，不包括滤波
'''

def main():
    outPath = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\output"
    paramspath = r"D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings"

    inputCSV = os.path.join(outPath, 'Dataset022_Lung.csv')
    outputFilepath = os.path.join(outPath, 'radiomics_022_features_Primitive.csv')
    progress_filename = os.path.join(outPath, 'pyrad_log_2.txt')
    params = os.path.join(paramspath, 'CT_lung_Primitive.yaml')

    # 配置日志记录
    rLogger = logging.getLogger('radiomics')
    handler = logging.FileHandler(filename=progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)
    logger = rLogger.getChild('batch')
    radiomics.setVerbosity(logging.INFO)

    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Loading CSV')

    flists = []
    try:
        with open(inputCSV, 'r') as inFile:
            cr = csv.DictReader(inFile, lineterminator='\n')
            flists = [row for row in cr]
    except Exception:
        logger.error('Failed to read CSV', exc_info=True)

    logger.info('CSV loaded')
    logger.info('Cases: %d', len(flists))

    if os.path.isfile(params):
        with open(params, 'r', encoding='utf-8') as file:
            params_dict = yaml.safe_load(file)
        extractor = featureextractor.RadiomicsFeatureExtractor(params_dict)
    else:
        settings = {'binWidth': 25, 'resampledPixelSpacing': None, 'interpolator': sitk.sitkBSpline, 'enableCExtensions': True}
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        #使用 featureextractor.RadiomicsFeatureExtractor 类初始化了一个特征提取器 extractor来根据参数setting提取特征。

    logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
    logger.info('Enabled features: %s', extractor.enabledFeatures)
    logger.info('Settings: %s', extractor.settings)

    # 在开始提取特征之前，初始化CSV文件并写入表头
    headers = None
    with open(outputFilepath, 'w', newline='') as outputFile:
        writer = csv.writer(outputFile, lineterminator='\n')
        # 表头将在处理第一个图像时确定并写入

    for idx, entry in enumerate(flists, start=1):
        logger.info("(%d/%d) Processing case (Image: %s, Mask: %s)", idx, len(flists), entry['Image'], entry['Label'])

        imageFilepath = entry['Image']
        maskFilepath = entry['Label']
        label = entry.get('Label', None)

        if str(label).isdigit():
            label = int(label)
        else:
            label = None

        if imageFilepath and maskFilepath:
            featureVector = collections.OrderedDict(entry)
            featureVector['Image'] = os.path.basename(imageFilepath)
            featureVector['Mask'] = os.path.basename(maskFilepath)

            try:
                featureVector.update(extractor.execute(imageFilepath, maskFilepath, label))

                if headers is None:
                    headers = list(featureVector.keys())
                    with open(outputFilepath, 'a', newline='') as outputFile:
                        writer = csv.writer(outputFile, lineterminator='\n')
                        writer.writerow(headers)

                row = [featureVector.get(h, "N/A") for h in headers]
                with open(outputFilepath, 'a', newline='') as outputFile:
                    writer = csv.writer(outputFile, lineterminator='\n')
                    writer.writerow(row)
            except Exception:
                logger.error('Feature extraction failed', exc_info=True)

if __name__ == '__main__':
    main()


