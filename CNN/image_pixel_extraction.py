image = open("image.png", "rb")
data = image.read()
image.close()

print(type(data))