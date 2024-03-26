''
'''
# 这是一个简单的脚本，用于检查参数文件是否有效。
# 使用参数文件的位置作为命令行参数运行它（即“python testParams.py PATH/TO/PARAMFILE”）。
# 如果成功，将打印一条消息，显示指定的自定义参数。如果验证失败，将打印一条错误消息，指定验证错误的原因。
'''

import sys

import pykwalify.core

from radiomics import getParameterValidationFiles

def main(paramsFile):
  schemaFile, schemaFuncs = getParameterValidationFiles()

  c = pykwalify.core.Core(source_file=paramsFile, schema_files=[schemaFile], extensions=[schemaFuncs])
  try:
    params = c.validate()
    print('参数验证成功！\n\n'
          '###Enabled Features###\n%s\n'
          '###Enabled Image Types###\n%s\n'
          '###Settings###\n%s' % (params['featureClass'], params['imageType'], params['setting']))
  except Exception as e:
    print('参数验证失败！\n%s' % e.message)


if __name__ == '__main__' and len(sys.argv) > 1:
    main(sys.argv[1])
