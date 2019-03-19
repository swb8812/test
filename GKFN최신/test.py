import GKFN
import ft

data = ft.readData()

# 선택된 E, tau 값을 이용하여 데이터를 재구성합니다.
E = 6
tau = 9
P = 1 # P는 몇일 뒤 값을 예측할 지 설정 해주는 parameter입니다. 예를 들어 하루 뒤 값을 예측하는 것이면 P = 1

# train set 및 test set을 구성합니다.
dataX, dataY  = ft.extracting(tau, E, P, data)
trX = dataX[:-92]
teX = dataX[-92:]
trY = dataY[:-92]
teY = dataY[-92:]

# parameter를 설정하고 학습을 시킵니다.
alpha = 0.25
loop = 5
Kernel_Num = 150

GKFN.GKFN(trX, trY, teX, teY, alpha, loop, Kernel_Num)

