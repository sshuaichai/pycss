# batchprocessing.py
批处理
# 参数文件路径
params = os.path.join(outPath, 'exampleSettings', 'Params.yaml')
# 输出路径为指定路径下
outPath = r''
# 输入CSV文件路径
inputCSV = os.path.join(outPath, 'testCases.csv')  
# 输出CSV文件路径
outputFilepath = os.path.join(outPath, 'radiomics_features.csv') 

# 输入CSV文件
从一个CSV文件中读取图像和掩模文件的路径，这些文件用于特征提取。
testCases.csv:
```angular2html
ID,Image,Mask
brain1,../data/brain1_image.nrrd,../data/brain1_label.nrrd
brain2,../data/brain2_image.nrrd,../data/brain2_label.nrrd
breast1,../data/breast1_image.nrrd,../data/breast1_label.nrrd
lung1,../data/lung1_image.nrrd,../data/lung1_label.nrrd
lung2,../data/lung2_image.nrrd,../data/lung2_label.nrrd
```

# 进度日志文件
pyrad_log.txt
```angular2html
INFO:radiomics.batch: pyradiomics version: v3.1.0
INFO:radiomics.batch: Loading CSV
INFO:radiomics.batch: Loading Done
INFO:radiomics.batch: Patients: 5
INFO:radiomics.featureextractor: Loading parameter file exampleSettings\Params.yaml
INFO:radiomics.batch: Enabled input images types: {'Original': {}}
INFO:radiomics.batch: Enabled features: {'shape': None, 'firstorder': [], 'glcm': ['Autocorrelation', 'JointAverage', 'ClusterProminence', 'ClusterShade', 'ClusterTendency', 'Contrast', 'Correlation', 'DifferenceAverage', 'DifferenceEntropy', 'DifferenceVariance', 'JointEnergy', 'JointEntropy', 'Imc1', 'Imc2', 'Idm', 'Idmn', 'Id', 'Idn', 'InverseVariance', 'MaximumProbability', 'SumEntropy', 'SumSquares'], 'glrlm': None, 'glszm': None, 'gldm': None}
INFO:radiomics.batch: Current settings: {'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 1, 'additionalInfo': True, 'binWidth': 25, 'weightingNorm': None}
INFO:radiomics.batch: (1/3) Processing Patient (Image: ../data/brain1_image.nrrd, Mask: ../data/brain1_label.nrrd)
INFO:radiomics.featureextractor: Calculating features with label: 1
INFO:radiomics.featureextractor: Loading image and mask
INFO:radiomics.featureextractor: Computing shape
INFO:radiomics.featureextractor: Adding image type "Original" with custom settings: {}
INFO:radiomics.featureextractor: Calculating features for original image
INFO:radiomics.featureextractor: Computing firstorder
INFO:radiomics.featureextractor: Computing glcm
INFO:radiomics.featureextractor: Computing glrlm
INFO:radiomics.featureextractor: Computing glszm
INFO:radiomics.featureextractor: Computing gldm
INFO:radiomics.batch: (2/3) Processing Patient (Image: ../data/brain2_image.nrrd, Mask: ../data/brain2_label.nrrd)
INFO:radiomics.featureextractor: Calculating features with label: 1
INFO:radiomics.featureextractor: Loading image and mask
INFO:radiomics.featureextractor: Computing shape
INFO:radiomics.featureextractor: Adding image type "Original" with custom settings: {}
INFO:radiomics.featureextractor: Calculating features for original image
INFO:radiomics.featureextractor: Computing firstorder
INFO:radiomics.featureextractor: Computing glcm
INFO:radiomics.featureextractor: Computing glrlm
INFO:radiomics.featureextractor: Computing glszm
INFO:radiomics.featureextractor: Computing gldm
INFO:radiomics.batch: (3/3) Processing Patient (Image: ../data/breast1_image.nrrd, Mask: ../data/breast1_label.nrrd)
INFO:radiomics.featureextractor: Calculating features with label: 1
INFO:radiomics.featureextractor: Loading image and mask
INFO:radiomics.featureextractor: Computing shape
INFO:radiomics.featureextractor: Adding image type "Original" with custom settings: {}
INFO:radiomics.featureextractor: Calculating features for original image
INFO:radiomics.featureextractor: Computing firstorder
INFO:radiomics.featureextractor: Computing glcm
INFO:radiomics.featureextractor: Computing glrlm
INFO:radiomics.featureextractor: Computing glszm
INFO:radiomics.featureextractor: Computing gldm
INFO:radiomics.batch: (4/3) Processing Patient (Image: ../data/lung1_image.nrrd, Mask: ../data/lung1_label.nrrd)
INFO:radiomics.featureextractor: Calculating features with label: 1
INFO:radiomics.featureextractor: Loading image and mask
INFO:radiomics.featureextractor: Computing shape
INFO:radiomics.featureextractor: Adding image type "Original" with custom settings: {}
INFO:radiomics.featureextractor: Calculating features for original image
INFO:radiomics.featureextractor: Computing firstorder
INFO:radiomics.featureextractor: Computing glcm
INFO:radiomics.featureextractor: Computing glrlm
INFO:radiomics.featureextractor: Computing glszm
INFO:radiomics.featureextractor: Computing gldm
INFO:radiomics.batch: (5/3) Processing Patient (Image: ../data/lung2_image.nrrd, Mask: ../data/lung2_label.nrrd)
INFO:radiomics.featureextractor: Calculating features with label: 1
INFO:radiomics.featureextractor: Loading image and mask
INFO:radiomics.featureextractor: Computing shape
INFO:radiomics.featureextractor: Adding image type "Original" with custom settings: {}
INFO:radiomics.featureextractor: Calculating features for original image
INFO:radiomics.featureextractor: Computing firstorder
INFO:radiomics.featureextractor: Computing glcm
INFO:radiomics.featureextractor: Computing glrlm
INFO:radiomics.featureextractor: Computing glszm
INFO:radiomics.featureextractor: Computing gldm
INFO:radiomics.batch: Extraction complete, writing CSV
INFO:radiomics.batch: CSV writing complete
```

# 输出CSV文件
每对图像和掩模，执行特征提取，并将提取的特征写入到CSV文件中
radiomics_features.csv
```angular2html
ID,Image,Mask,diagnostics_Versions_PyRadiomics,diagnostics_Versions_Numpy,diagnostics_Versions_SimpleITK,diagnostics_Versions_PyWavelet,diagnostics_Versions_Python,diagnostics_Configuration_Settings,diagnostics_Configuration_EnabledImageTypes,diagnostics_Image-original_Hash,diagnostics_Image-original_Dimensionality,diagnostics_Image-original_Spacing,diagnostics_Image-original_Size,diagnostics_Image-original_Mean,diagnostics_Image-original_Minimum,diagnostics_Image-original_Maximum,diagnostics_Mask-original_Hash,diagnostics_Mask-original_Spacing,diagnostics_Mask-original_Size,diagnostics_Mask-original_BoundingBox,diagnostics_Mask-original_VoxelNum,diagnostics_Mask-original_VolumeNum,diagnostics_Mask-original_CenterOfMassIndex,diagnostics_Mask-original_CenterOfMass,original_shape_Elongation,original_shape_Flatness,original_shape_LeastAxisLength,original_shape_MajorAxisLength,original_shape_Maximum2DDiameterColumn,original_shape_Maximum2DDiameterRow,original_shape_Maximum2DDiameterSlice,original_shape_Maximum3DDiameter,original_shape_MeshVolume,original_shape_MinorAxisLength,original_shape_Sphericity,original_shape_SurfaceArea,original_shape_SurfaceVolumeRatio,original_shape_VoxelVolume,original_firstorder_10Percentile,original_firstorder_90Percentile,original_firstorder_Energy,original_firstorder_Entropy,original_firstorder_InterquartileRange,original_firstorder_Kurtosis,original_firstorder_Maximum,original_firstorder_MeanAbsoluteDeviation,original_firstorder_Mean,original_firstorder_Median,original_firstorder_Minimum,original_firstorder_Range,original_firstorder_RobustMeanAbsoluteDeviation,original_firstorder_RootMeanSquared,original_firstorder_Skewness,original_firstorder_TotalEnergy,original_firstorder_Uniformity,original_firstorder_Variance,original_glcm_Autocorrelation,original_glcm_JointAverage,original_glcm_ClusterProminence,original_glcm_ClusterShade,original_glcm_ClusterTendency,original_glcm_Contrast,original_glcm_Correlation,original_glcm_DifferenceAverage,original_glcm_DifferenceEntropy,original_glcm_DifferenceVariance,original_glcm_JointEnergy,original_glcm_JointEntropy,original_glcm_Imc1,original_glcm_Imc2,original_glcm_Idm,original_glcm_Idmn,original_glcm_Id,original_glcm_Idn,original_glcm_InverseVariance,original_glcm_MaximumProbability,original_glcm_SumEntropy,original_glcm_SumSquares,original_glrlm_GrayLevelNonUniformity,original_glrlm_GrayLevelNonUniformityNormalized,original_glrlm_GrayLevelVariance,original_glrlm_HighGrayLevelRunEmphasis,original_glrlm_LongRunEmphasis,original_glrlm_LongRunHighGrayLevelEmphasis,original_glrlm_LongRunLowGrayLevelEmphasis,original_glrlm_LowGrayLevelRunEmphasis,original_glrlm_RunEntropy,original_glrlm_RunLengthNonUniformity,original_glrlm_RunLengthNonUniformityNormalized,original_glrlm_RunPercentage,original_glrlm_RunVariance,original_glrlm_ShortRunEmphasis,original_glrlm_ShortRunHighGrayLevelEmphasis,original_glrlm_ShortRunLowGrayLevelEmphasis,original_glszm_GrayLevelNonUniformity,original_glszm_GrayLevelNonUniformityNormalized,original_glszm_GrayLevelVariance,original_glszm_HighGrayLevelZoneEmphasis,original_glszm_LargeAreaEmphasis,original_glszm_LargeAreaHighGrayLevelEmphasis,original_glszm_LargeAreaLowGrayLevelEmphasis,original_glszm_LowGrayLevelZoneEmphasis,original_glszm_SizeZoneNonUniformity,original_glszm_SizeZoneNonUniformityNormalized,original_glszm_SmallAreaEmphasis,original_glszm_SmallAreaHighGrayLevelEmphasis,original_glszm_SmallAreaLowGrayLevelEmphasis,original_glszm_ZoneEntropy,original_glszm_ZonePercentage,original_glszm_ZoneVariance,original_gldm_DependenceEntropy,original_gldm_DependenceNonUniformity,original_gldm_DependenceNonUniformityNormalized,original_gldm_DependenceVariance,original_gldm_GrayLevelNonUniformity,original_gldm_GrayLevelVariance,original_gldm_HighGrayLevelEmphasis,original_gldm_LargeDependenceEmphasis,original_gldm_LargeDependenceHighGrayLevelEmphasis,original_gldm_LargeDependenceLowGrayLevelEmphasis,original_gldm_LowGrayLevelEmphasis,original_gldm_SmallDependenceEmphasis,original_gldm_SmallDependenceHighGrayLevelEmphasis,original_gldm_SmallDependenceLowGrayLevelEmphasis
brain1,brain1_image.nrrd,brain1_label.nrrd,v3.1.0,1.26.3,2.3.1,1.3.0,3.9.12,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 1, 'additionalInfo': True, 'binWidth': 25, 'weightingNorm': None}",{'Original': {}},5c9ce3ca174f0f8324aa4d277e0fef82dc5ac566,3D,"(0.7812499999999999, 0.7812499999999999, 6.499999999999998)","(256, 256, 25)",385.6564080810547,0.0,3057.0,9dc2c3137b31fd872997d92c9a92d5178126d9d3,"(0.7812499999999999, 0.7812499999999999, 6.499999999999998)","(256, 256, 25)","(162, 84, 11, 47, 70, 7)",4137,2,"(186.98549673676578, 106.3562968334542, 14.38917089678511)","(46.47304432559825, 16.518518098863908, 15.529610829103234)",0.5621171627174117,0.4610597534658259,28.584423185376494,61.99722046980878,49.490854979101925,65.88905951721043,53.59397776919529,69.60099030590368,16147.51180013021,34.84970166685475,0.4798234536231475,6438.821603779402,0.3987500788652454,16412.658691406243,632.0,1044.4,2918821481.0,4.601935553903786,253.0,2.1807729393860265,1266.0,133.44726195252767,825.2354363065023,812.0,468.0,798.0,103.00138343026683,839.9646448180755,0.27565085908587594,11579797135.314934,0.045156963555862184,24527.07920837261,289.5436994017259,16.55380772442751,27995.937591943148,19.605083427286676,108.73139325453903,47.492125114429776,0.3917522006696661,5.284468789866316,3.74406097806642,16.65563705027098,0.002893149242988865,8.799696270248813,-0.09438938808738298,0.6942249020670357,0.20022255640475703,0.961402169623227,0.28722572382985156,0.8726052157397169,0.19881884197093194,0.007352392266290182,5.354241321485615,39.05587959224222,175.6351923150419,0.04514123814981055,39.118151021979244,281.066493908972,1.2268440382584342,341.2865790983503,0.010601170478748765,0.008600397891661503,4.915038003159503,3500.0432315746298,0.8950494659480998,0.9404064632491029,0.08479457789590625,0.9559391731405504,268.9741798411307,0.008229766244155428,82.38716577540107,0.044057307901283996,40.60313992393263,288.6235294117647,13.615508021390374,3514.7614973262034,0.12723841553344326,0.009100942027706215,747.5967914438503,0.3997843804512568,0.6564478999587141,193.438051925864,0.006416982055097711,6.5082149861981895,0.4520183708000967,8.721239097486347,6.885019899269458,936.6601401982113,0.22641047623838803,2.1619872286911965,186.8143582306019,39.19271419906397,280.4065748126662,8.661590524534686,2335.0519700265895,0.07650590736710827,0.00860027409479837,0.37960167130711403,110.30563945728201,0.0035453562622343696
brain2,brain2_image.nrrd,brain2_label.nrrd,v3.1.0,1.26.3,2.3.1,1.3.0,3.9.12,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 1, 'additionalInfo': True, 'binWidth': 25, 'weightingNorm': None}",{'Original': {}},f2b8fbc4d5d1da08a1a70e79a301f8a830139438,3D,"(0.7812499999999999, 0.7812499999999999, 6.499999999999998)","(256, 256, 23)",236.20217696480128,0.0,1523.0,b41049c71633e194bee4891750392b72eabd8800,"(0.7812499999999999, 0.7812499999999999, 6.499999999999998)","(256, 256, 23)","(205, 155, 8, 20, 15, 3)",453,1,"(215.21192052980132, 161.45474613686534, 8.589403973509933)","(58.134312913907266, -22.717773471183904, -6.868874935124886)",0.7407691177548195,0.6188162226844611,11.180247958180273,18.067153943831176,23.59325330603009,20.728928495281654,16.312978920172732,24.444532236269108,1736.0178629557292,13.383589687312327,0.7513044980804245,929.7601508652821,0.5355706128981187,1797.180175781249,305.2,532.6,73903770.0,3.8206001230555597,97.0,4.674716196193526,729.0,73.55426906227311,390.7858719646799,375.0,8.0,721.0,44.01867695920074,403.90961876693444,0.43863725063173664,293197329.7119139,0.09596557655853302,10429.38240525513,276.93903753785196,16.542459664591465,7743.448102798923,127.13809223840286,42.317452064295665,29.293839050703014,0.2105277382606347,3.9826729595606403,3.339049784655726,11.757829127963502,0.01469411462568149,6.813008822502754,-0.2410315210432108,0.918128262963891,0.261778488335105,0.9709270908607235,0.34588791642687816,0.8920419265947748,0.25338046923906604,0.03898727381744979,4.384121367195153,17.902822778749666,38.4475791133819,0.0919964760550404,17.628601560571752,281.65691437161036,1.37117405444786,369.81991828393836,0.009392897201128767,0.007606402393319139,4.209760406206385,365.0749453470162,0.8663631725348966,0.9181524876889114,0.1503662075285962,0.9406569413053879,266.8619287697709,0.0073222575251349,9.63803680981595,0.05912906018291995,27.850427189581843,319.9877300613497,35.3680981595092,7639.0245398773,0.1846593516154854,0.011743556352087885,69.1840490797546,0.4244420188942,0.6770632920785222,218.87922881968453,0.009883953613125893,5.671105706980072,0.3598233995584989,27.644472881930067,6.027854291423979,77.86092715231788,0.1718784263848077,3.153584881754699,43.47240618101545,16.831308568337644,278.15673289183223,12.938189845474614,3072.9955849889625,0.06477163542005925,0.007352909129511827,0.3143642190135804,98.73745327196367,0.003919611286864095
breast1,breast1_image.nrrd,breast1_label.nrrd,v3.1.0,1.26.3,2.3.1,1.3.0,3.9.12,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 1, 'additionalInfo': True, 'binWidth': 25, 'weightingNorm': None}",{'Original': {}},016951a8f9e8e5de21092d9d62b77262f92e04a5,3D,"(0.664062, 0.664062, 2.1)","(45, 120, 21)",63.380440917107585,6.0,248.0,c45601d81c387b7c809948af79ebf173738c0b02,"(0.664062, 0.664062, 2.1)","(45, 120, 21)","(21, 64, 8, 9, 12, 3)",143,1,"(24.58041958041958, 69.27272727272727, 9.132867132867133)","(-100.22011741258741, -2.1432681818181933, 28.629020979021007)",0.6999838102754058,0.684224676154207,5.289714502170175,7.730961315078332,7.1214084629446734,8.24091097816352,8.399793710266938,9.081018608331778,124.09130483210153,5.411547758420293,0.6959037135176507,172.88707947619218,1.3932247687306738,132.42579545515324,110.2,151.8,2495543.0,1.4490921348001997,30.0,1.8044408150083193,162.0,13.93975255513717,131.12587412587413,130.0,102.0,60.0,11.083091863105961,132.10355653936668,0.04182797313132072,2311008.859213562,0.38901657782776666,257.35478507506485,3.9663379593499664,1.9585401105995826,3.2994515259044612,-0.016711211805404247,1.1696127286855562,0.6712186625058774,0.2649029994159988,0.5711662030283523,1.1846976792586774,0.32725642574315206,0.17361171785231078,2.7724168824971636,-0.08733050328813872,0.4199223123375775,0.7244221444335763,0.9374959395714517,0.7310923083987446,0.8622110722167882,0.48362030098551795,0.2966562800200133,2.062198798134395,0.4602078477978584,40.786454258354595,0.388358506415894,0.4907437930759212,3.422543729347743,2.847472964887773,10.561165915290028,1.3863094937410545,0.5538046292958577,2.428993490104397,65.4722188360389,0.5908466884847003,0.7380311995696611,0.6632564055207943,0.783377717977769,2.5526507882555447,0.4635767830900796,2.2,0.44,0.64,3.2,1272.6,4671.6,520.85,0.6722222222222223,1.0,0.2,0.051098362147739426,0.05486014011949099,0.05066182292061157,2.321928094887361,0.03496503496503497,454.64,4.661728196602649,13.251748251748252,0.09266956819404372,11.04826641889579,55.62937062937063,0.4749376497628246,3.5804195804195804,72.06293706293707,295.9160839160839,27.0475912975913,0.518065268065268,0.03121098627290894,0.08194786618439001,0.022342990461955925
lung1,lung1_image.nrrd,lung1_label.nrrd,v3.1.0,1.26.3,2.3.1,1.3.0,3.9.12,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 1, 'additionalInfo': True, 'binWidth': 25, 'weightingNorm': None}",{'Original': {}},34dca4200809a5e76c702d6b9503d958093057a3,3D,"(0.5703125, 0.5703125, 5.0)","(512, 512, 48)",-577.9148241678873,-1024.0,3071.0,c9ca19983d5fe7d3e85fbfd751d979ab999caf97,"(0.5703125, 0.5703125, 5.0)","(512, 512, 48)","(206, 347, 32, 24, 26, 3)",837,1,"(217.6726403823178, 358.5125448028674, 33.200716845878134)","(-21.85857228195937, -120.53581429211468, -611.4964157706091)",0.7187910312752431,0.5143357681770736,8.936318223606332,17.374483317150457,19.555665988802964,17.557494838387395,16.627323762644803,21.344946720202447,1330.976079305013,12.488622781409086,0.7480376581937057,782.2414579991738,0.5877201477637612,1361.1978149414062,-245.39999999999998,71.0,16291991.0,4.0208349271393775,198.0,2.695927095819359,106.0,105.09444751337841,-63.90800477897252,-31.0,-506.0,612.0,81.58090534979424,139.5161077616851,-0.733665953604179,26495367.44354248,0.07442664462743855,15380.51125014096,411.4164748228452,20.045124840449564,9732.694395990255,-345.7133672514899,58.74756668430427,20.713449304070664,0.47061361659594525,3.2166030915302346,3.187524501755002,9.381995812782842,0.01791827061861308,6.9328289964103265,-0.1733118679446957,0.8187663824873835,0.34435017672215545,0.9725823938178468,0.41736196376573204,0.8996307073762659,0.2786971666959305,0.08912560596848712,4.635501946477711,19.86525399709374,48.26523799762741,0.06601836750150518,24.661240952561574,362.39939518560374,1.7567901942027135,758.7811249911574,0.007773425022858408,0.006164928954270328,4.555631762141003,602.3643646725302,0.8196942123519452,0.8688539656281594,0.38206181267492273,0.9200285720284883,322.21283053397593,0.00597933358883647,18.29530201342282,0.06139363091752624,21.55781271113914,262.744966442953,93.21812080536913,51136.969798657716,0.1848753071287851,0.010725735940860114,138.7248322147651,0.4655195711904869,0.7094782224171242,170.30796183011958,0.009509531665402186,5.514483642327986,0.35603345280764637,85.32918562226926,6.550399891550477,120.97610513739546,0.14453537053452264,18.002914773562637,62.29510155316607,24.733677910384273,383.9199522102748,37.44922341696535,20425.035842293906,0.07556248355632293,0.005605862273240823,0.3181860034269303,84.05116858601934,0.003630174681129408
lung2,lung2_image.nrrd,lung2_label.nrrd,v3.1.0,1.26.3,2.3.1,1.3.0,3.9.12,"{'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 1, 'additionalInfo': True, 'binWidth': 25, 'weightingNorm': None}",{'Original': {}},14f57fd04838eb8c9cca2a0dd871d29971585975,3D,"(0.6269531, 0.6269531, 5.0)","(512, 512, 61)",-433.28809775680793,-1024.0,1671.0,295d315a562eb6b4ce7d5f592b6cd3558f1d5cf6,"(0.6269531, 0.6269531, 5.0)","(512, 512, 61)","(318, 333, 15, 87, 66, 11)",24644,1,"(355.8619542282097, 364.6708326570362, 20.70094140561597)","(52.608755375434185, -95.86849098608988, -256.49529297192004)",0.7433464635239115,0.5692892353827336,31.75180297332847,55.77446577218645,65.30307864957555,62.10262893946419,55.802347997267205,68.98247103025058,48292.521604579546,41.45975188669025,0.6724099300593408,9537.544879912664,0.1974952759353992,48434.108762463955,-102.0,60.70000000000073,233096758.0,2.6529019354413617,30.0,12.884706923906494,297.0,59.581897780966464,7.602093815938971,41.0,-840.0,1137.0,19.66978625210457,97.25512951766174,-3.0222132456965567,458116934.3105721,0.2815372119563662,9400.768387110824,1250.784627888118,35.30865346044656,13254.85338280221,-498.73190629686474,28.339974380216194,12.115468558259831,0.36123715466021683,1.5501077045805824,2.1805601671536725,9.471117969228834,0.14996278839692623,4.329838764966286,-0.12908095573211506,0.5428625433960625,0.6520977174114816,0.9947250027676501,0.6748882805280818,0.9708450560250022,0.3513643575798707,0.32422085412558943,3.1809315369280786,10.113860734619008,2548.8156623544014,0.17104695463266578,23.257155860667826,1165.5187022255527,6.809541279184898,8739.459798442862,0.005534637986269019,0.0010625570649908174,4.66603787302322,7485.99904620261,0.5017706103253707,0.5875326183311901,3.4307119998894917,0.733442155651029,818.1416749470084,0.0008557235289409869,158.6329365079365,0.052457981649449904,36.25726627456538,829.786044973545,53609.46626984127,70294988.98941799,40.908036032945354,0.0018944881662849348,1486.9775132275133,0.49172536813079143,0.7289780684420905,569.9826397432703,0.001565588007755414,6.025803729100555,0.12270735270248337,53543.05239389995,6.503742395704024,1074.54609641292,0.043602746973418274,53.9169284521668,6938.203051452686,15.086611671909008,1226.7050803441,191.3726667748742,250067.02475247523,0.146818467095096,0.0009344850373972612,0.11508039120847749,97.93058453415173,0.0002186069898830078
```
-----------------------------------------------------------------------
# batchprocessing_parallel.py
批处理_
 # 参数文件地址
PARAMS = os.path.join(ROOT, 'exampleSettings', 'Params.yaml')
ROOT = os.getcwd() 获取当前脚本所在的目录，并将其赋值给变量 ROOT。可保存文件在当前目录下。
# 输入CSV文件路径
INPUTCSV = os.path.join(ROOT, 'testCases.csv') 
# 输出CSV文件路径
OUTPUTCSV = os.path.join(ROOT, 'results.csv')  

# 输出日志文件
log.txt
```angular2html
I: (Main) radiomics.batch: pyradiomics version: v3.1.0
I: (Main) radiomics.batch: Loading CSV...
I: (Main) radiomics.batch: Loaded 5 jobs
I: (Main) radiomics.batch: Creating temporary output directory D:\zhuomian\pyradiomics\pyradiomics-master\examples\_TEMP
I: (Main) radiomics.batch: Starting parralel pool with 15 workers out of 16 CPUs
I: (1) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (2) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (1) radiomics.featureextractor: Calculating features with label: 1
I: (1) radiomics.featureextractor: Loading image and mask
I: (2) radiomics.featureextractor: Calculating features with label: 1
I: (2) radiomics.featureextractor: Loading image and mask
I: (3) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (4) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (5) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (3) radiomics.featureextractor: Calculating features with label: 1
I: (3) radiomics.featureextractor: Loading image and mask
I: (4) radiomics.featureextractor: Calculating features with label: 1
I: (4) radiomics.featureextractor: Loading image and mask
I: (5) radiomics.featureextractor: Calculating features with label: 1
I: (5) radiomics.featureextractor: Loading image and mask
I: (3) radiomics.featureextractor: Computing shape
I: (3) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (3) radiomics.featureextractor: Calculating features for original image
I: (3) radiomics.featureextractor: Computing firstorder
I: (3) radiomics.featureextractor: Computing glcm
I: (3) radiomics.featureextractor: Computing glrlm
I: (3) radiomics.featureextractor: Computing glszm
I: (3) radiomics.featureextractor: Computing gldm
I: (1) radiomics.featureextractor: Computing shape
I: (3) radiomics.batch: Patient 3 read by N-A processed in 0:00:00.143888
I: (2) radiomics.featureextractor: Computing shape
I: (2) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (2) radiomics.featureextractor: Calculating features for original image
I: (1) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (1) radiomics.featureextractor: Calculating features for original image
I: (2) radiomics.featureextractor: Computing firstorder
I: (2) radiomics.featureextractor: Computing glcm
I: (1) radiomics.featureextractor: Computing firstorder
I: (1) radiomics.featureextractor: Computing glcm

I: (2) radiomics.featureextractor: Computing glszm
I: (2) radiomics.featureextractor: Computing gldm
I: (2) radiomics.batch: Patient 2 read by N-A processed in 0:00:00.257561
I: (1) radiomics.featureextractor: Computing glrlm
I: (1) radiomics.featureextractor: Computing glszm
I: (1) radiomics.featureextractor: Computing gldm
I: (1) radiomics.batch: Patient 1 read by N-A processed in 0:00:00.298520
I: (4) radiomics.featureextractor: Computing shape
I: (4) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (4) radiomics.featureextractor: Calculating features for original image
I: (4) radiomics.featureextractor: Computing firstorder
I: (4) radiomics.featureextractor: Computing glcm
I: (4) radiomics.featureextractor: Computing glrlm
I: (4) radiomics.featureextractor: Computing glszm
I: (4) radiomics.featureextractor: Computing gldm
I: (4) radiomics.batch: Patient 4 read by N-A processed in 0:00:00.861275
I: (5) radiomics.featureextractor: Computing shape
I: (5) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (5) radiomics.featureextractor: Calculating features for original image
I: (5) radiomics.featureextractor: Computing firstorder
I: (5) radiomics.featureextractor: Computing glcm
I: (5) radiomics.featureextractor: Computing glrlm
I: (5) radiomics.featureextractor: Computing glszm
I: (5) radiomics.featureextractor: Computing gldm
I: (5) radiomics.batch: Patient 5 read by N-A processed in 0:00:01.222294
I: (Main) radiomics.batch: Removing temporary directory D:\zhuomian\pyradiomics\pyradiomics-master\examples\_TEMP (contains individual case results files)
I: (Main) radiomics.batch: pyradiomics version: v3.1.0
I: (Main) radiomics.batch: Loading CSV...
I: (Main) radiomics.batch: Loaded 5 jobs
I: (Main) radiomics.batch: Creating temporary output directory D:\zhuomian\pyradiomics\pyradiomics-master\examples\_TEMP
I: (Main) radiomics.batch: Starting parralel pool with 15 workers out of 16 CPUs
I: (1) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (2) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (3) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (4) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (1) radiomics.featureextractor: Calculating features with label: 1
I: (1) radiomics.featureextractor: Loading image and mask
I: (2) radiomics.featureextractor: Calculating features with label: 1
I: (2) radiomics.featureextractor: Loading image and mask
I: (3) radiomics.featureextractor: Calculating features with label: 1
I: (3) radiomics.featureextractor: Loading image and mask
I: (5) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (4) radiomics.featureextractor: Calculating features with label: 1
I: (4) radiomics.featureextractor: Loading image and mask
I: (5) radiomics.featureextractor: Calculating features with label: 1
I: (5) radiomics.featureextractor: Loading image and mask
I: (4) radiomics.featureextractor: Computing shape
I: (3) radiomics.featureextractor: Computing shape
I: (1) radiomics.featureextractor: Computing shape
I: (4) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (4) radiomics.featureextractor: Calculating features for original image
I: (3) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (3) radiomics.featureextractor: Calculating features for original image
I: (4) radiomics.featureextractor: Computing firstorder
I: (3) radiomics.featureextractor: Computing firstorder
I: (3) radiomics.featureextractor: Computing glcm
I: (4) radiomics.featureextractor: Computing glcm
I: (2) radiomics.featureextractor: Computing shape
I: (3) radiomics.featureextractor: Computing glrlm
I: (4) radiomics.featureextractor: Computing glrlm
I: (1) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (1) radiomics.featureextractor: Calculating features for original image
I: (1) radiomics.featureextractor: Computing firstorder
I: (1) radiomics.featureextractor: Computing glcm
I: (1) radiomics.featureextractor: Computing glrlm
I: (2) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (2) radiomics.featureextractor: Calculating features for original image
I: (2) radiomics.featureextractor: Computing firstorder
I: (2) radiomics.featureextractor: Computing glcm
I: (2) radiomics.featureextractor: Computing glrlm
I: (3) radiomics.featureextractor: Computing glszm
I: (5) radiomics.featureextractor: Computing shape
I: (3) radiomics.featureextractor: Computing gldm
I: (4) radiomics.featureextractor: Computing glszm
I: (3) radiomics.batch: Patient 3 read by N-A processed in 0:00:06.422573
I: (4) radiomics.featureextractor: Computing gldm
I: (5) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (5) radiomics.featureextractor: Calculating features for original image
I: (5) radiomics.featureextractor: Computing firstorder
I: (4) radiomics.batch: Patient 4 read by N-A processed in 0:00:06.441708
I: (5) radiomics.featureextractor: Computing glcm
I: (1) radiomics.featureextractor: Computing glszm
I: (5) radiomics.featureextractor: Computing glrlm
I: (1) radiomics.featureextractor: Computing gldm
I: (2) radiomics.featureextractor: Computing glszm
I: (1) radiomics.batch: Patient 1 read by N-A processed in 0:00:06.579087
I: (2) radiomics.featureextractor: Computing gldm
I: (2) radiomics.batch: Patient 2 read by N-A processed in 0:00:06.618816
I: (5) radiomics.featureextractor: Computing glszm
I: (5) radiomics.featureextractor: Computing gldm
I: (5) radiomics.batch: Patient 5 read by N-A processed in 0:00:06.698961
I: (Main) radiomics.batch: Removing temporary directory D:\zhuomian\pyradiomics\pyradiomics-master\examples\_TEMP (contains individual case results files)
I: (Main) radiomics.batch: pyradiomics version: v3.1.0
I: (Main) radiomics.batch: Loading CSV...
I: (Main) radiomics.batch: Loaded 5 jobs
I: (Main) radiomics.batch: Creating temporary output directory D:\zhuomian\pyradiomics\pyradiomics-master\examples\_TEMP
I: (Main) radiomics.batch: Starting parralel pool with 15 workers out of 16 CPUs
I: (1) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (1) radiomics.featureextractor: Calculating features with label: 1
I: (1) radiomics.featureextractor: Loading image and mask
I: (2) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (3) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (2) radiomics.featureextractor: Calculating features with label: 1
I: (2) radiomics.featureextractor: Loading image and mask
I: (4) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (5) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (3) radiomics.featureextractor: Calculating features with label: 1
I: (3) radiomics.featureextractor: Loading image and mask
I: (5) radiomics.featureextractor: Calculating features with label: 1
I: (5) radiomics.featureextractor: Loading image and mask
I: (4) radiomics.featureextractor: Calculating features with label: 1
I: (4) radiomics.featureextractor: Loading image and mask
I: (3) radiomics.featureextractor: Computing shape
I: (3) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (3) radiomics.featureextractor: Calculating features for original image
I: (3) radiomics.featureextractor: Computing firstorder
I: (3) radiomics.featureextractor: Computing glcm
I: (3) radiomics.featureextractor: Computing glrlm
I: (3) radiomics.featureextractor: Computing glszm
I: (3) radiomics.featureextractor: Computing gldm
I: (3) radiomics.batch: Patient 3 read by N-A processed in 0:00:00.135301
I: (1) radiomics.featureextractor: Computing shape
I: (1) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (1) radiomics.featureextractor: Calculating features for original image
I: (2) radiomics.featureextractor: Computing shape
I: (1) radiomics.featureextractor: Computing firstorder
I: (1) radiomics.featureextractor: Computing glcm
I: (2) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (2) radiomics.featureextractor: Calculating features for original image
I: (2) radiomics.featureextractor: Computing firstorder
I: (1) radiomics.featureextractor: Computing glrlm
I: (2) radiomics.featureextractor: Computing glcm
I: (1) radiomics.featureextractor: Computing glszm
I: (1) radiomics.featureextractor: Computing gldm
I: (1) radiomics.batch: Patient 1 read by N-A processed in 0:00:00.314200
I: (2) radiomics.featureextractor: Computing glrlm
I: (2) radiomics.featureextractor: Computing glszm
I: (2) radiomics.featureextractor: Computing gldm
I: (2) radiomics.batch: Patient 2 read by N-A processed in 0:00:00.238445
I: (4) radiomics.featureextractor: Computing shape
I: (4) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (4) radiomics.featureextractor: Calculating features for original image
I: (4) radiomics.featureextractor: Computing firstorder
I: (4) radiomics.featureextractor: Computing glcm
I: (4) radiomics.featureextractor: Computing glrlm
I: (4) radiomics.featureextractor: Computing glszm
I: (4) radiomics.featureextractor: Computing gldm
I: (4) radiomics.batch: Patient 4 read by N-A processed in 0:00:00.830524
I: (5) radiomics.featureextractor: Computing shape
I: (5) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (5) radiomics.featureextractor: Calculating features for original image
I: (5) radiomics.featureextractor: Computing firstorder
I: (5) radiomics.featureextractor: Computing glcm
I: (5) radiomics.featureextractor: Computing glrlm
I: (5) radiomics.featureextractor: Computing glszm
I: (5) radiomics.featureextractor: Computing gldm
I: (5) radiomics.batch: Patient 5 read by N-A processed in 0:00:01.173138
I: (Main) radiomics.batch: Removing temporary directory D:\zhuomian\pyradiomics\pyradiomics-master\examples\_TEMP (contains individual case results files)
I: (Main) radiomics.batch: pyradiomics version: v3.1.0
I: (Main) radiomics.batch: Loading CSV...
I: (Main) radiomics.batch: Loaded 5 jobs
I: (Main) radiomics.batch: Creating temporary output directory D:\zhuomian\pyradiomics\pyradiomics-master\examples\_TEMP
I: (Main) radiomics.batch: Starting parralel pool with 15 workers out of 16 CPUs
I: (1) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (1) radiomics.featureextractor: Calculating features with label: 1
I: (1) radiomics.featureextractor: Loading image and mask
I: (2) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (3) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (4) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (5) radiomics.featureextractor: Loading parameter file D:\zhuomian\pyradiomics\pyradiomics-master\examples\exampleSettings\Params.yaml
I: (2) radiomics.featureextractor: Calculating features with label: 1
I: (2) radiomics.featureextractor: Loading image and mask
I: (3) radiomics.featureextractor: Calculating features with label: 1
I: (3) radiomics.featureextractor: Loading image and mask
I: (4) radiomics.featureextractor: Calculating features with label: 1
I: (4) radiomics.featureextractor: Loading image and mask
I: (5) radiomics.featureextractor: Calculating features with label: 1
I: (5) radiomics.featureextractor: Loading image and mask
I: (1) radiomics.featureextractor: Computing shape
I: (3) radiomics.featureextractor: Computing shape
I: (3) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (3) radiomics.featureextractor: Calculating features for original image
I: (3) radiomics.featureextractor: Computing firstorder
I: (3) radiomics.featureextractor: Computing glcm
I: (3) radiomics.featureextractor: Computing glrlm
I: (1) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (3) radiomics.featureextractor: Computing glszm
I: (1) radiomics.featureextractor: Calculating features for original image
I: (3) radiomics.featureextractor: Computing gldm
I: (3) radiomics.batch: Patient 3 read by N-A processed in 0:00:00.143203
I: (1) radiomics.featureextractor: Computing firstorder
I: (1) radiomics.featureextractor: Computing glcm
I: (1) radiomics.featureextractor: Computing glrlm
I: (1) radiomics.featureextractor: Computing glszm
I: (1) radiomics.featureextractor: Computing gldm
I: (1) radiomics.batch: Patient 1 read by N-A processed in 0:00:00.330659
I: (2) radiomics.featureextractor: Computing shape
I: (2) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (2) radiomics.featureextractor: Calculating features for original image
I: (2) radiomics.featureextractor: Computing firstorder
I: (2) radiomics.featureextractor: Computing glcm
I: (2) radiomics.featureextractor: Computing glrlm
I: (2) radiomics.featureextractor: Computing glszm
I: (2) radiomics.featureextractor: Computing gldm
I: (2) radiomics.batch: Patient 2 read by N-A processed in 0:00:00.220724
I: (4) radiomics.featureextractor: Computing shape
I: (4) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (4) radiomics.featureextractor: Calculating features for original image
I: (4) radiomics.featureextractor: Computing firstorder
I: (4) radiomics.featureextractor: Computing glcm
I: (4) radiomics.featureextractor: Computing glrlm
I: (4) radiomics.featureextractor: Computing glszm
I: (4) radiomics.featureextractor: Computing gldm
I: (4) radiomics.batch: Patient 4 read by N-A processed in 0:00:00.846779
I: (5) radiomics.featureextractor: Computing shape
I: (5) radiomics.featureextractor: Adding image type "Original" with custom settings: {}
I: (5) radiomics.featureextractor: Calculating features for original image
I: (5) radiomics.featureextractor: Computing firstorder
I: (5) radiomics.featureextractor: Computing glcm
I: (5) radiomics.featureextractor: Computing glrlm
I: (5) radiomics.featureextractor: Computing glszm
I: (5) radiomics.featureextractor: Computing gldm
I: (5) radiomics.batch: Patient 5 read by N-A processed in 0:00:01.219960
I: (Main) radiomics.batch: Removing temporary directory D:\zhuomian\pyradiomics\pyradiomics-master\examples\_TEMP (contains individual case results files)
```

