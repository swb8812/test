import numpy as np

############################### 함수

def extracting(tau, E, P, obj):
    in_put = []
    out_put = []
    a = tau * (E-1)
    for i in range(a, len(obj)-P):
        b=[]
        for j in range(i-a, i+tau, tau):
            b.append(obj[j])
        in_put.append(b)
        out_put.append(obj[i+P])
    return np.array(in_put), np.array(out_put)


def SM(input_data, output_data, E):
    sm = 0
    for i in range(len(input_data)):
        temp = []
        for j in range(len(input_data)):
            temp.append(Dist(input_data[i], input_data[j], E))

        temp = np.array(temp)

        nonzerotemp = temp[np.nonzero(temp)]
        nonzerotemp.sort()

        for j in range(len(temp)):
            if nonzerotemp[0] == temp[j]:
                idx = j

        minDist = temp[idx]

        sm += np.abs(output_data[i] - output_data[idx]) / minDist
    #        print(format("%f th step minDist: %f, minIndex: %f, sm: %f") % (i, minDist, idx, sm))

    sm = 1 - sm / len(input_data)

    return sm


def Dist(x1, x2, E):
    dist = 0
    for i in range(0, E):
        dist += np.square(x1[i] - x2[i])

    return np.sqrt(dist)

################################################################## 코드 시작

# train에 학습할 데이터를 리스트로 만들어주세요.
train = []
# P는 예측하려고 하는 날의 수 입니다. 예를 들면 내일의 값을 예측하려고 하면 P = 1, 3일 뒤를 예측하려고 하면 P = 3
P = 1
f = open('여기에 저장 위치 설정해주세요', 'w')
for E in range(4, 11):
    for tau in range(1, 21):
        temp = []
        a1, a2 = extracting(tau, E, P, train)
        sm = SM(a1, a2, E)
        temp.append(tau)
        temp.append(E)
        temp.append(sm)
        print(format("E: %f, tau: %f sm: %f") % (E, tau, sm))
        f.write(format("E: %f, tau: %f sm: %f") % (E, tau, sm) + '\n')
f.close()
