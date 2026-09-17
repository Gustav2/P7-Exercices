# Import the library you would like to use
# NB -- first you have to install it: pip install requests
import requests
import numpy as np

# Signup
requests.post("http://localhost:8080/signup", json={"name":"Bjarki"})

###################################Exercise 1####################################

# Define exercise number
ex = 1

# Get exercise data
r = requests.get("http://localhost:8080/ex/{}".format(ex), headers={"x-data":"True"})
data = r.json()["data"]
print("Data received: {}".format(data))

# Solve the problem

res = []
for x in data:
 if x <= 30:
   res.append(x)

# Send the solution½
r = requests.post("http://localhost:8080/ex/{}".format(ex), json={"data":res})

# Print the result
print(r.json())
print("Result: {}".format(res))


######################################exercise 2####################################
ex = 2

# Get exercise data
r = requests.get("http://localhost:8080/ex/{}".format(ex), headers={"x-data":"True"})
data = r.json()["data"]
print("Data received: {}".format(data))

last = data[len(data) - 1]
for i, x in enumerate(data):
    data[i] = x + last


print("Result: {}".format(data))
r = requests.post("http://localhost:8080/ex/{}".format(ex), json={"data":data})

# Print the result

print(r.json())
print("Result: {}".format(data))
######################################Exercise 3####################################
ex = 3
r = requests.get("http://localhost:8080/ex/{}".format(ex), headers={"x-data":"True"})
data = r.json()["data"]
print("Data received: {}".format(data))

data.sort()
result = []
for i, x in enumerate(data):
    x = "5" + x[1:]
    result.append(x)


r = requests.post("http://localhost:8080/ex/{}".format(ex), json={"data":result})

# Print the result
print("Result: {}".format(result))
print(r.json())



#####################################Exercise 4####################################
ex = 4
r = requests.get("http://localhost:8080/ex/{}".format(ex), headers={"x-data":"True"})
data = r.json()["data"]
print("Data received: {}".format(data))

text = data["text"]
result =""

for i in range(5):
    result += text


r = requests.post("http://localhost:8080/ex/{}".format(ex), json={"data":result})

# Print the result
print("Result: {}".format(result))
print(r.json())


######################################Exercise 5####################################
ex = 5
r = requests.get("http://localhost:8080/ex/{}".format(ex), headers={"x-data":"True"})
data = r.json()["data"]
print("Data received: {}".format(data))


num = 0
result = []
testnum = 1000
while num < 5:
    
    if testnum % data == 0:
        num += 1
        result.append(testnum)
    testnum += 1



r = requests.post("http://localhost:8080/ex/{}".format(ex), json={"data":result})

# Print the result
print("Result: {}".format(result))
print(r.json())



