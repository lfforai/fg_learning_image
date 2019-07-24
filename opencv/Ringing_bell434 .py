import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

y_down=np.zeros(1000,dtype=float)
y_up=np.zeros(1000,dtype=float)
x=np.linspace(-128,128,2000)
y=np.array(0.035*np.sin(3.1415926*x*0.035)/(3.1415926*x*0.035))
y_down=np.append(y_down,y)
y_down=np.append(y_down,y_up)
y=y_down
x=np.linspace(-256,256,4000)
plt.figure()
plt.plot(x,y)
plt.show()
y_last=[]
i=0
for e in y:
    y_last.append(e*np.power(-1,i))
    i=i+1
sinc_fft=fft(y_last)
y_last=[]
i=0
for e in sinc_fft.real:
    y_last.append(e*np.power(-1,i))
    i=i+1

y=np.array(y_last)
y1=y*(512)/4000
abc=[]
for e in  y1:
    if abs(e)>0.001:
       abc.append(e)
print(abc.__len__())
x=list(range(abc.__len__()))
plt.plot(x,abc)
plt.show()

# x=[1.0,2.0,4.0,4.0]
# x_fft=fft(x)
# print(x_fft.real)
# print(x_fft.imag)
# w=ifft(x_fft)
# print(w.real)
# print(w.imag)


