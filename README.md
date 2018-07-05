# 从指定训练数据中生成朴素贝叶斯分类器并对数据进行预测
1.运行环境：
Python3、
Numpy

2.从指定训练数据中生成朴素贝叶斯分类器并预测:

从终端进入项目文件夹，输入：python3，运行Python解释器，在Python解释器中输入以下命令：

    import naive_bayes 
    # 生成分类器，可参考该方法从指定文件生成分类器
    myBayes=naive_bayes.testA()
    # 预测单条数据
    myBayes.predictData([特征1,特征2,特征3...特征n])
    # 从测试数据中检测分类器错误率
    myBayes.predict("adult.test.txt")

预测数据类型可包含：连续值、离散值以及缺失值。

3.参数说明：

    # 指定需要读取的训练数据文件名，应保证该文件与decisionTree.py在同一文件夹下
	self.fileName
	# 指定训练数据使用的分割符号
	self.fileSplitStr
	# 指定训练数据各特征值名称
	self.attribute_names
	# 指定训练数据各特征值类型：continuous表示该特征值为连续型；discrete表示该特征值为离散型
	self.attribute_types
	# 指定训练数据中缺省值的表示符号
	self.unknownMark
  
4.提供的测试数据中包含连续值、离散值以及缺失值，共32561条训练数据，数据详细信息参考：(http://archive.ics.uci.edu/ml/datasets/Adult)
