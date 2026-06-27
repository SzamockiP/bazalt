import lumapy
import time

eng = lumapy.Engine()

eng.init()

def fun():
    for x in range(10):
        time.sleep(1)
        print("Dziala", x)
    eng.stop()

eng.run(fun)
print("koniec")