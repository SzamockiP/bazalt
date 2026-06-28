import lumapy
import time

engine = lumapy.Engine()

engine.init(1024, 720, "test")

@engine.onError
def error(msg):
    print(msg)

def fun():
    i = 0
    while engine.running():
        if engine.isKeyPressed(ord("A")):
            i += 1
            engine.log(str(i))

        if engine.isKeyPressed(ord("X")):
            engine.stop()


engine.run(fun)
print("koniec")