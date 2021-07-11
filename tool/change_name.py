import os
import os.path

rootdir = '/root/data/covid19/test_rename/subset4/434/'

a=0
for name in range(11, 96):
    print(name)
    newname = str(a) + '.jpg'
    a = a+1
    os.rename(rootdir+str(name)+'.jpg', rootdir+newname)
