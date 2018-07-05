from numpy import *

class Bayes():
	"""docstring for Bayes"""
	def __init__(self):
		# 数据文件名
		self.fileName = ""
		# 缺失值标记
		self.unknownMark = ""
		# 文件数据分割符
		self.fileSplitStr = ""
		# 训练集
		self.trainSet = []
		# 属性名称
		self.attribute_names = []
		# 属性数据类型'continuous','discrete'
		self.attribute_types = []
		# 属性索引，用于预测
		self.predictAttrIndexDic = {}

		self.attr_Dic = {}
		self.bayesAttr = {}


	# 从指定文件中读取文件，参数：文件名，分割符
	def readTrainFile(self,fileName,splitStr):
		self.fileName = fileName
		self.fileSplitStr = splitStr
		dataSet = []
		attr_length = len(self.attribute_names) + 1
		fr = open(fileName)
		index = 0
		for line in fr.readlines():
			newLine = line.strip().split(splitStr)
			if len(newLine) == 1:
				# 如果分割后只有一条数据，则提示分割符是否正确，并退出
				print("分割符不正确！")
				# 文件名、分割符置空
				self.fileName = ""
				self.fileSplitStr = ""
				break
			index += 1
			attributes = [item.strip() for item in newLine]
			if len(attributes) != attr_length:
				print("第",index,"条数据格式可能不正确")
				continue
			dataSet.append(attributes)
		return dataSet
		pass

	def scanTrainSet(self):
		attr_Dic = {}
		for data in self.trainSet:
			for attr_index in range(len(self.attribute_names)):
				# 如果当前数据该属性值缺失
				if data[attr_index] == self.unknownMark:
					continue
				attr = attr_Dic.get(self.attribute_names[attr_index],{})
				totalValid = attr.get("totalValid",{})
				classify = data[-1]
				totalValid[classify] = totalValid.get(classify,0) + 1
				classifyDic = attr.get("classifyValues",{})
				# 离散
				if self.attribute_types[attr_index] == "discrete":
					attr["dataType"] = "discrete"
					validTypesDic = classifyDic.get(data[attr_index],{})
					validTypesDic[classify] = validTypesDic.get(classify,0) + 1
					classifyDic[data[attr_index]] = validTypesDic
				else:
					# 连续
					attr["dataType"] = "continuous"
					validList = classifyDic.get(classify,[])
					validList.append(float(data[attr_index]))
					classifyDic[classify] = validList
				attr["classifyValues"] = classifyDic
				attr["totalValid"] = totalValid
				attr_Dic[self.attribute_names[attr_index]] = attr


		for key in attr_Dic.keys():
			isContinuous = attr_Dic[key]["dataType"] == "continuous"
			classifyDic = attr_Dic[key]["classifyValues"]
			if isContinuous:
				allList = []
				for c_key in classifyDic.keys():
					allList.extend(classifyDic[c_key])
				attr_Dic[key][self.unknownMark] = mean(allList)
			else:
				countDic = {}
				for c_key in classifyDic.keys():
					countDic[c_key] = sum(list(classifyDic[c_key].values()))
					pass
				value = ""
				maxCount = 0
				for count_key in countDic.keys():
					if countDic[count_key] > maxCount:
						maxCount = countDic[count_key]
						value = count_key
				attr_Dic[key][self.unknownMark] = value

		self.attr_Dic = attr_Dic

	def updateBayesAttr(self):
		bayesAttr = {}
		for key in self.attr_Dic.keys():
			item = self.attr_Dic[key]
			radioDic = {}
			for classify_key in item["totalValid"].keys():
					classifyRadio = {}
					if item["dataType"] == "discrete":
						valuesNum = len(list(item["classifyValues"].keys()))
						for value_key in item["classifyValues"].keys():
							# 拉普拉斯平滑
							classifyRadio[value_key] = float(item["classifyValues"][value_key].get(classify_key,0) + 1)/\
							float(item["totalValid"][classify_key] + valuesNum)
					else:
						valueList = item["classifyValues"][classify_key]
						classifyRadio["u"] = mean(valueList)
						classifyRadio["o_2"] = var(valueList)
					radioDic[classify_key] = classifyRadio
			bayesAttr[key] = radioDic
		self.bayesAttr = bayesAttr
		pass

	def train(self):
		if not self.checkPropoty():
			return
		self.trainSet = self.readTrainFile(self.fileName,self.fileSplitStr)
		self.create_attributes_indexDic()
		self.scanTrainSet()
		self.updateBayesAttr()
		pass

	def create_attributes_indexDic(self):
		index = 0
		for attr_name in self.attribute_names:
			self.predictAttrIndexDic[attr_name] = index
			index += 1
			pass
		pass

	# 检测必要属性是否为空
	def checkPropoty(self):
		if self.checkEmpty(self.fileName):
			print("fileName为空")
			return False
		if self.checkEmpty(self.unknownMark):
			print("unknownMark为空")
			return False
		if self.checkEmpty(self.fileSplitStr):
			print("fileSplitStr为空")
			return False
		if self.checkEmpty(self.attribute_names):
			print("attribute_names为空")
			return False
		if self.checkEmpty(self.attribute_types):
			print("attribute_types为空")
			return False
		return True
		pass

	# 检测是否为空
	def checkEmpty(self,obj):
		if len(obj) == 0:
			return True
		return False
		pass

	def predictData(self,data):
		if len(data) != len(self.attribute_names):
			print("参数数量不符合")
			pass
		radioDic = {}
		for attr_index in range(len(data)):
			attr_name = self.attribute_names[attr_index]
			attr_value = data[attr_index]
			attr_type = self.attribute_types[attr_index]
			isContinuous = (attr_type == "continuous")
			if attr_value == self.unknownMark:
				attr_value = self.attr_Dic[attr_name][self.unknownMark]
				pass
			for classify_key in self.bayesAttr[attr_name].keys():
				radio_list = radioDic.get(classify_key,[])
				if isContinuous:
					u = float(self.bayesAttr[attr_name][classify_key]["u"])
					o_2 = float(self.bayesAttr[attr_name][classify_key]["o_2"])
					value_ = float(attr_value)
					radio = (1/pow(2*pi*o_2,0.5))*exp(-((value_-u)**2)/(2*o_2))
					# if radio == float(0):
					# 	print("isContinuous:",isContinuous,"jjjj:")
					# 	print("value_:",value_,"u:",u,"o_2:",o_2,"llll:",(1/pow(2*pi*o_2,0.5))*exp(-((value_-u)**2)/(2*o_2)))
				else:
					radio = self.bayesAttr[attr_name][classify_key][attr_value]

				radio_list.append(radio)
				radioDic[classify_key] = radio_list
		maxRadio = -inf
		predict = ""
		# print("radioDic:",radioDic)
		for key in radioDic.keys():
			if float(0) in radioDic[key]:
				# print("radioDic:",radioDic[key],"data:",data)
				continue
			r_list = log(radioDic[key])
			radio = sum(r_list)
			# print("radio:",radio)
			if radio > maxRadio:
				maxRadio = radio
				predict = key
		return predict

	def predict(self,fileName):
		predictDataSet = []
		trueResult = []
		fr = open(fileName)
		for line in fr.readlines():
			newLine = line.strip().split(",")
			attributes = [item.strip() for item in newLine]
			# if not self.unknownMark in attributes:
			# 	continue
			trueResult.append(attributes[-1].strip("."))
			del(attributes[-1])
			predictDataSet.append(attributes)
		errorNum = 0
		totalNum = len(predictDataSet)
		errorIndex = []
		for index in range(len(predictDataSet)):
			data = predictDataSet[index]
			if len(data) != len(self.attribute_names):
				print("第",index,"条数据，参数数量不符合")
				continue
				pass
			result = self.predictData(data)
			if result != trueResult[index]:
				print("index:",index,"trueResult:",trueResult[index],"result:",result)
				errorNum += 1
				errorIndex.append(index)
		e = errorNum/totalNum
		return e

def testA():
	adult = Bayes()
	# 指定需要读取的训练数据文件名，应保证该文件与decisionTree.py在同一文件夹下
	adult.fileName = "adult.data.txt"
	# 指定训练数据使用的分割符号
	adult.fileSplitStr = ","
	# 指定训练数据各特征值名称
	adult.attribute_names = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship',\
			    'race','sex','capital-gain','capital-loss','hours-per-week','native-country']
	# 指定训练数据各特征值类型：continuous表示该特征值为连续型；discrete表示该特征值为离散型
	adult.attribute_types = ['continuous','discrete','continuous','discrete','continuous','discrete','discrete','discrete',\
			    'discrete','discrete','continuous','continuous','continuous','discrete']
	# 指定训练数据中缺省值的表示符号
	adult.unknownMark = "?"

	adult.train()

	return adult
	pass
