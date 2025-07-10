
import matplotlib.pyplot as plt
import numpy as np
# 2D
# x1= np.zeros(100)
# for i in range(100):
#     x1[i] = 0.1*i-5
# x2=x1
# x3= np.sin(x1)
#
# plt.figure()
# plt.plot(x1,x2,'b')
# plt.plot(x1,x3,'c')
# plt.grid('on')
# plt.show()

# 3D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()
ax = plt.axes(projection='3d')

u = np.linspace(0, 2 * np.pi, 100)

x = u
y = u
z = u
y1 = np.sin(x)
z1 = 1
ax.plot3D(x, y, z,'c')
ax.plot3D(x, y1, z1,'b')
plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.outer(np.linspace(-2,2,30), np.ones(30))
# y = x.copy().T
# z =1+0 *np.cos(x **2+ y **2)
# z1= np.sin(y)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_surface(x, y, z,color='b')
# ax.plot_surface(x, y, z1,color='c')
# ax.set_title('Surface  ')
# plt.show()



