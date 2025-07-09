# 課題 1-1_矢田目和弥.py
from math import *
import numpy as np
import matplotlib.pyplot as plt
import Ref


f=2.0  
lamda=0.3/f  
k=2*pi/lamda  
d=np.arange(0, 100, 0.1)  # 連続区間を0.1単位で区切る離散化に加え，ベクトルを用いることでループを1000回回すより高速化

Pt=0.0  
Gt=0.0  
Gr=0.0 
Gt=sqrt(10**(Gt/10))  
Gr=sqrt(10**(Gr/10))  
zt=3.0  
zr=1.5  
Er=5.0-1j*60*0.005*lamda  
E0=1  


d0=np.sqrt(d**2+(zt-zr)**2)  # 送信点から，受信点dまでの直接パスの距離
dg=np.sqrt(d**2+(zt+zr)**2)  


theta=np.arccos((zt+zr)/dg)  
Rv, Rh=Ref.Ref(theta,Er)  # (d, yr, 0)上の，各dにおける反射角度から反射係数を計算


Ed=E0*Gt*Gr*(lamda/4/pi/d0)  
Egv=E0*Gt*Gr*(lamda/4/pi/dg)*Rv*np.exp(-1j*k*(dg-d0))


Erec_v=Ed+Egv
L_v=-20*np.log10(np.abs(Erec_v))  # d毎の受信電界強度をdB表示


Prec_v=Pt-L_v; 


fig = plt.figure()
ax = fig.add_subplot(111)
fig.suptitle("Distance Characteristics of 2 path model")
ax.plot(d, Prec_v, color='b', label="V-pol.")
ax.set_xlabel("Distance d (m)")
ax.set_ylabel("Received Power (dBm)")
ax.set_xlim(0, 100)
ax.set_ylim(-90, -30)
ax.grid()
plt.show()

#
# End of file