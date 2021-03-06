from __future__ import print_function
from six.moves.urllib import request
import zipfile
import os

url = "http://cs231n.stanford.edu/coco_captioning.zip"

file_name = url.split('/')[-1]
u = request.urlopen(url)
f = open(file_name, 'wb')
meta = u.info()
file_size = int(meta.get("Content-Length"))
print("Downloading: %s Bytes: %s" % (file_name, file_size))

file_size_dl = 0
block_sz = 1048576
while True:
    buffer = u.read(block_sz)
    if not buffer:
        break

    file_size_dl += len(buffer)
    f.write(buffer)
    status = "%d  [%3.2f%%]\r" % (file_size_dl, file_size_dl * 100. / file_size)
    print(status,end='')

f.close()

print('Extracting: %s' % file_name)
t =  zipfile.ZipFile(file_name, "r")  
t.extractall('.')  
t.close()  

os.remove(file_name)

url = "http://cs231n.stanford.edu/imagenet_val_25.npz"

file_name = url.split('/')[-1]
u = request.urlopen(url)
f = open(file_name, 'wb')
meta = u.info()
file_size = int(meta.get("Content-Length"))
print("Downloading: %s Bytes: %s" % (file_name, file_size))

file_size_dl = 0
block_sz = 1048576
while True:
    buffer = u.read(block_sz)
    if not buffer:
        break

    file_size_dl += len(buffer)
    f.write(buffer)
    status = "%d  [%3.2f%%]\r" % (file_size_dl, file_size_dl * 100. / file_size)
    print(status,end='')

f.close()

url = "http://cs231n.stanford.edu/squeezenet_tf.zip"

file_name = url.split('/')[-1]
u = request.urlopen(url)
f = open(file_name, 'wb')
meta = u.info()
file_size = int(meta.get("Content-Length"))
print("Downloading: %s Bytes: %s" % (file_name, file_size))

file_size_dl = 0
block_sz = 1048576
while True:
    buffer = u.read(block_sz)
    if not buffer:
        break

    file_size_dl += len(buffer)
    f.write(buffer)
    status = "%d  [%3.2f%%]\r" % (file_size_dl, file_size_dl * 100. / file_size)
    print(status,end='')

f.close()

print('Extracting: %s' % file_name)
t =  zipfile.ZipFile(file_name, "r")  
t.extractall('.')  
t.close()  

os.remove(file_name)