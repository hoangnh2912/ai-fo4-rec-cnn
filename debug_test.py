from start_server import readb64

f = open("loi.txt", "r")
data = f.read()
data = data.split('\n')
data = list(filter(lambda x: len(x) > 100, data))

for id, x in enumerate(data):
    readb64(x, 'cache/' + str(id) + ".png")
