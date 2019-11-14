import xlrd
from xlwt import Workbook

data ='data/glcm.xls'
label = 'data/Train_label.xlsx'
wb = xlrd.open_workbook(filename=label)#打开文件
wb2=xlrd.open_workbook(filename=data)#特征
shee1 = wb.sheet_by_index(0)
shee_1 = wb2.sheet_by_index(0)
print(shee1.name,shee1.nrows,shee1.ncols)
print(shee_1.name,shee_1.nrows,shee_1.ncols)
flag = 1
info = dict()
wb = Workbook()
sheet2 = wb.add_sheet('Sheet 1')
for i in range(1,744):
    row = shee1.row_values(i)
    key = row[0][1:][:-1]
    # print(key)
    info[key] = row[1][1:][:-1]+","+row[2][1:][:-1]
# print(info)
for j in range(1,shee_1.nrows):
    r = shee_1.row_values(j)
    imgLabel = info[r[0]]
    imgLabels = imgLabel.split(",")
    print(imgLabels)
    sheet2.write(flag, 0, r[0])
    sheet2.write(flag,1,imgLabels[0])
    sheet2.write(flag,2,imgLabels[1])
    flag+=1
wb.save("data/label.xls")

