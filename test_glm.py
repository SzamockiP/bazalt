import glm
import struct

m = glm.mat4(1, 2, 3, 4,
             5, 6, 7, 8,
             9, 10, 11, 12,
             13, 14, 15, 16)

# m is initialized with arguments in column-major order according to GLM docs.
# So column 0 is [1, 2, 3, 4].

b1 = bytes(m)
b2 = bytes(glm.transpose(m))

data = []
m_t = glm.transpose(m)
for i in range(4):
    for j in range(4):
        data.append(m_t[i][j])

b3 = struct.pack(f'{len(data)}f', *data)

print("bytes(m)       :", struct.unpack('16f', b1))
print("bytes(m_t)     :", struct.unpack('16f', b2))
print("data loops     :", struct.unpack('16f', b3))
