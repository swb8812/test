import ft

"""Data Load + data structure 결정"""
# Data Load
# 이 데이터 대신 본인 데이터를 쓰시면 됩니다.
data =ft.readData()

# data에 대하여 E, tau에 따른 smoothness 값 측정
# 이 함수를 실행시키면 폴더에 sm.txt라는 파일이 생성됩니다.
ft.smoothnessMeasure(data)



